//! Serializer / deserializer for bridge-side optimizer knowledge.
//!
//! Ports `rpython/jit/metainterp/optimizeopt/bridgeopt.py`:
//!
//! * `serialize_optimizer_knowledge` (bridgeopt.py:63-122) writes the
//!   known-class bitfield + heap field/array triples + loopinvariant
//!   call-result tuples onto a guard's `rd_numb` stream when finishing
//!   resume data.
//! * `deserialize_optimizer_knowledge` (bridgeopt.py:124-185) reads those
//!   sections back at bridge-compile time and applies the facts directly
//!   onto the bridge optimizer (`Optimizer::make_constant_class`,
//!   `import_heap_knowledge`, `import_loopinvariant_knowledge`). RPython
//!   has no separate "BridgeKnowledge" struct or per-guard pass — facts
//!   are written into the standard optimizer state and consumed by the
//!   existing OptIntBounds / OptHeap / OptVirtualize passes.
//!
//! `decoded_box_to_opref` is a small helper for folding a typed
//! `Const{Int,Float,Ptr}` from `crate::resume::decode_box` back into the
//! optimizer's constant pool.

use majit_ir::OpRef;

use crate::optimizeopt::OptContext;

/// bridgeopt.py:124-185 deserialize_optimizer_knowledge.
///
/// Read optimizer knowledge from the guard's rd_numb and apply it
/// directly to the optimizer passes. RPython parity: the function
/// takes the optimizer and applies knowledge inline, never returning
/// an intermediate struct.
/// bridgeopt.py:124 signature:
/// deserialize_optimizer_knowledge(optimizer, resumestorage, frontend_boxes, liveboxes)
///
/// bridgeopt.py:63-122 `serialize_optimizer_knowledge(optimizer,
/// numb_state, liveboxes, liveboxes_from_env, memo)`.
///
/// Emits three serialized sections on every guard (RPython emits zeros
/// when the optheap/optrewrite caches are empty; the deserializer relies
/// on the sections always being present):
///
/// 1. known-class bitfield per Ref livebox (bridgeopt.py:74-90)
/// 2. heap field + array item triples (bridgeopt.py:92-108)
/// 3. loopinvariant call results (bridgeopt.py:113-122)
///
/// RPython splits the memo-side wrapper (`_add_optimizer_sections`,
/// resume.py:570-574) from the serialize core (`serialize_optimizer_knowledge`,
/// bridgeopt.py:63-122). pyre keeps the same split: this free function
/// carries the core, and `ResumeDataLoopMemo::_add_optimizer_sections`
/// forwards.
pub fn serialize_optimizer_knowledge(
    memo: &mut crate::resume::ResumeDataLoopMemo,
    numb_state: &mut crate::resume::NumberingState,
    liveboxes: &[Option<OpRef>],
    new_liveboxes: &crate::resume::LiveboxMap,
    env: &dyn majit_ir::BoxEnv,
    optimizer_knowledge: Option<&crate::resume::OptimizerKnowledgeForResume>,
) {
    // bridgeopt.py:64-67 `available_boxes = {}` followed by
    // `available_boxes[box] = None` — RPython uses a dict as a
    // membership set (values are always None). Rust mirrors the dict
    // shape with `HashMap<OpRef, ()>` rather than `HashSet<OpRef>` to
    // keep structural parity with the RPython data-structure choice.
    let available_boxes: std::collections::HashMap<OpRef, ()> = liveboxes
        .iter()
        .filter_map(|opt| *opt)
        .filter(|opref| numb_state.liveboxes.contains_key(*opref))
        .map(|opref| (opref, ()))
        .collect();

    // bridgeopt.py:74-88: known classes bitfield
    // RPython: for each livebox, call getptrinfo(box).get_known_class(cpu).
    // The actual class pointer is recovered at deserialization time
    // via cpu.cls_of_box(frontend_boxes[i]).
    //
    // RPython Box.type parity: bridgeopt.py:77 uses `box.type != "r"`,
    // where `box.type` is intrinsic/immutable. Pyre reads the same
    // type that `finish()` stores in `numb_state.livebox_types` (this
    // map feeds `fail_arg_types` / `livebox_types` on the deserialize
    // side — see bridgeopt.rs below). If we queried `env.get_type()`
    // here instead, a livebox whose OptContext-side type differs from
    // its numbering-time type would cause serialize/deserialize to
    // disagree on which Ref-typed slots get a bitfield bit, producing
    // an out-of-bounds rd_numb read in `deserialize_optimizer_knowledge`
    // when super-instruction GEN widens the live register set.
    let mut bitfield: i32 = 0;
    let mut shifts = 0;
    for livebox in liveboxes {
        if let Some(opref) = livebox {
            let livebox_tp = numb_state
                .livebox_types
                .get(opref)
                .copied()
                .unwrap_or_else(|| env.get_type(*opref));
            if livebox_tp != majit_ir::Type::Ref {
                continue;
            }
            bitfield <<= 1;
            // bridgeopt.py:79-80: info = getptrinfo(box)
            // known_class = info is not None and info.get_known_class(cpu) is not None
            if env.has_known_class(*opref) {
                bitfield |= 1;
            }
            shifts += 1;
            if shifts == 6 {
                numb_state.append_int(bitfield as i64);
                bitfield = 0;
                shifts = 0;
            }
        }
    }
    if shifts > 0 {
        numb_state.append_int((bitfield << (6 - shifts)) as i64);
    }

    // bridgeopt.py:92-122: heap knowledge
    let Some(knowledge) = optimizer_knowledge else {
        // bridgeopt.py:109-111,121-122: no optheap/optrewrite → zeros
        numb_state.append_int(0); // struct fields count
        numb_state.append_int(0); // array items count
        numb_state.append_int(0); // loopinvariant count
        return;
    };
    // bridgeopt.py:93: triples_struct = optimizer.optheap.serialize_optheap(available_boxes)
    let filtered_fields: Vec<(OpRef, i32, OpRef)> = knowledge
        .heap_fields
        .iter()
        .copied()
        .filter(|&(obj, _, val)| {
            let obj_ok = env.is_const(obj) || available_boxes.contains_key(&obj);
            let val_ok = env.is_const(val) || available_boxes.contains_key(&val);
            obj_ok && val_ok
        })
        .collect();
    numb_state.append_int(filtered_fields.len() as i64);
    for (obj, descr_idx, val) in &filtered_fields {
        let obj_tag = memo._gettagged(*obj, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(obj_tag as i32);
        numb_state.append_int(*descr_idx as i64);
        let val_tag = memo._gettagged(*val, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(val_tag as i32);
    }
    // bridgeopt.py:102-108: array items
    let filtered_arrayitems: Vec<(OpRef, i64, i32, OpRef)> = knowledge
        .heap_arrayitems
        .iter()
        .copied()
        .filter(|&(obj, _, _, val)| {
            let obj_ok = env.is_const(obj) || available_boxes.contains_key(&obj);
            let val_ok = env.is_const(val) || available_boxes.contains_key(&val);
            obj_ok && val_ok
        })
        .collect();
    numb_state.append_int(filtered_arrayitems.len() as i64);
    for (obj, index, descr_idx, val) in &filtered_arrayitems {
        let obj_tag = memo._gettagged(*obj, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(obj_tag as i32);
        // bridgeopt.py:106 numb_state.append_int(index) — pass the original
        // index unchanged; resumecode.py:90-93 enforces SHORT range on the
        // i64 value, panicking instead of silently wrapping a too-large
        // index into an i32.
        numb_state.append_int(*index);
        numb_state.append_int(*descr_idx as i64);
        let val_tag = memo._gettagged(*val, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(val_tag as i32);
    }

    // bridgeopt.py:113-122: loopinvariant results
    let filtered_loopinvariant: Vec<(i64, OpRef)> = knowledge
        .loopinvariant_results
        .iter()
        .copied()
        .filter(|&(_, result)| env.is_const(result) || available_boxes.contains_key(&result))
        .collect();
    numb_state.append_int(filtered_loopinvariant.len() as i64);
    for (const_ptr, result) in &filtered_loopinvariant {
        let const_tag = memo.getconst_int(*const_ptr);
        numb_state.writer.append_short(const_tag as i32);
        let result_tag = memo._gettagged(*result, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(result_tag as i32);
    }
}

/// `frontend_boxes`: runtime values from guard failure (RPython Box objects
///   with concrete references). Used by cls_of_box to read vtable.
/// `cls_of_box`: model.py:199-201 cpu.cls_of_box(box) — reads typeptr from
///   a runtime Ref object. Returns the class pointer as i64.
pub fn deserialize_optimizer_knowledge(
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    frontend_boxes: &[i64],
    liveboxes: &[OpRef],
    livebox_types: &[majit_ir::Type],
    all_descrs: &[majit_ir::descr::DescrRef],
    cls_of_box: Option<fn(i64) -> i64>,
    optimizer: &mut super::optimizer::Optimizer,
    ctx: &mut OptContext,
) {
    use crate::resume::{DecodedBox, decode_box};
    use majit_ir::resumecode::Reader;

    let mut reader = Reader::new(rd_numb);
    // bridgeopt.py:126: assert len(frontend_boxes) == len(liveboxes)
    assert!(
        frontend_boxes.len() == liveboxes.len(),
        "frontend_boxes.len()={} != liveboxes.len()={}",
        frontend_boxes.len(),
        liveboxes.len(),
    );

    // bridgeopt.py:130-131: skip resume section
    let startcount = reader.next_item();
    reader.jump((startcount - 1) as usize);

    // bridgeopt.py:133-146: class knowledge
    let mut bitfield: i32 = 0;
    let mut mask: i32 = 0;
    for (i, &livebox) in liveboxes.iter().enumerate() {
        // bridgeopt.py:135 reads `box.type` (intrinsic on the Box).
        // pyre's parallel side table must cover `liveboxes`.
        let tp = livebox_types.get(i).copied().unwrap_or_else(|| {
            panic!(
                "missing livebox_types[{}] (liveboxes.len()={}): \
                 RPython bridgeopt.py:135 reads box.type intrinsically; \
                 pyre's parallel array must match liveboxes length",
                i,
                liveboxes.len()
            )
        });
        if tp != majit_ir::Type::Ref {
            continue;
        }
        if mask == 0 {
            bitfield = reader.next_item();
            mask = 0b100000;
        }
        let class_known = (bitfield & mask) != 0;
        mask >>= 1;
        if class_known {
            // bridgeopt.py:145-146:
            //   cls = optimizer.cpu.cls_of_box(frontend_boxes[i])
            //   optimizer.make_constant_class(box, cls)
            // RPython's type system guarantees frontend_boxes[i] is a valid
            // GcRef when box.type == "r" and class_known is set. Our raw i64
            // encoding requires a nonnull check (RPython's box.nonnull()
            // equivalent, info.py:763).
            if let Some(cls_fn) = cls_of_box {
                let raw_ref = frontend_boxes[i];
                if raw_ref != 0 {
                    let cls = cls_fn(raw_ref);
                    super::optimizer::Optimizer::make_constant_class(ctx, livebox, cls, true);
                }
            }
        }
    }

    // bridgeopt.py:148-158: heap knowledge (struct fields)
    let length = reader.next_item();
    let mut result_struct = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let box1 = decode_box(tagged, rd_consts, liveboxes);
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        // bridgeopt.py:155: descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        let opref1 = decoded_box_to_opref(&box1, ctx);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        result_struct.push((opref1, descr.clone(), opref2));
    }
    // bridgeopt.py:159-169: heap knowledge (array items)
    let length = reader.next_item();
    let mut result_array = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let box1 = decode_box(tagged, rd_consts, liveboxes);
        let index = reader.next_item() as i64;
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        // bridgeopt.py:166: descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        let opref1 = decoded_box_to_opref(&box1, ctx);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        result_array.push((opref1, index, descr.clone(), opref2));
    }
    // bridgeopt.py:170-171: optimizer.optheap.deserialize_optheap(...)
    if !result_struct.is_empty() || !result_array.is_empty() {
        optimizer.import_heap_knowledge(&result_struct, &result_array, ctx);
    }

    // bridgeopt.py:173-185: call_loopinvariant knowledge
    let length = reader.next_item();
    let mut result_loopinvariant = Vec::new();
    for _ in 0..length {
        let tagged1 = reader.next_item() as i16;
        let const_box = decode_box(tagged1, rd_consts, liveboxes);
        // bridgeopt.py:179-180: assert isinstance(const, ConstInt); i = const.getint()
        let DecodedBox::Const(majit_ir::Const::Int(const_int)) = const_box else {
            panic!(
                "bridgeopt: loopinvariant entry must be ConstInt, got {:?}",
                const_box
            );
        };
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        // bridgeopt.py:183: result_loopinvariant.append((i, box))
        // No sentinel check — ConstInt(0) is a valid func_ptr value.
        result_loopinvariant.push((const_int, opref2));
    }
    // bridgeopt.py:184-185: optimizer.optrewrite.deserialize_optrewrite(...)
    if !result_loopinvariant.is_empty() {
        optimizer.import_loopinvariant_knowledge(&result_loopinvariant);
    }
}

/// Convert a DecodedBox to an OpRef for the bridge optimizer context.
///
/// RPython's deserialize path passes Const/Box objects directly. In majit,
/// constants must be registered in the optimizer's context to get an OpRef.
fn decoded_box_to_opref(decoded: &crate::resume::DecodedBox, ctx: &mut OptContext) -> OpRef {
    use crate::resume::DecodedBox;
    use majit_ir::Const;
    match decoded {
        DecodedBox::LiveBox(opref) => *opref,
        DecodedBox::Const(Const::Int(v)) => ctx.make_constant_int(*v),
        DecodedBox::Const(Const::Ref(r)) => ctx.make_constant_ref(*r),
        DecodedBox::Const(Const::Float(f)) => ctx.make_constant_float(*f),
    }
}
