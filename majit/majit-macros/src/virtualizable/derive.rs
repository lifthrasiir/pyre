//! `#[derive(VirtualizableSym)]` and `#[derive(VirtualizableMeta)]` implementation.
//!
//! Recognizes `#[vable(...)]` attributes on struct fields to generate
//! virtualizable-aware methods without changing the struct layout.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, Ident, Meta};

// ═══════════════════════════════════════════════════════════════
// #[vable(...)] attribute parsing
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VableRole {
    /// Frame pointer: `#[vable(frame)]`
    Frame,
    /// Inputarg scalar: included in jump/fail args. `#[vable(inputarg)]`
    Inputarg,
    /// Info-only field: heap layout only, not in jump/fail args. `#[vable(info_only)]`
    InfoOnly,
    /// Array base index: `#[vable(array_base)]`
    ArrayBase,
    /// Symbolic locals: `#[vable(locals)]`
    Locals,
    /// Symbolic stack: `#[vable(stack)]`
    Stack,
    /// Local types: `#[vable(local_types)]`
    LocalTypes,
    /// Stack types: `#[vable(stack_types)]`
    StackTypes,
    /// Number of locals: `#[vable(nlocals)]`
    Nlocals,
    /// Valuestackdepth: `#[vable(valuestackdepth)]`
    Valuestackdepth,
}

struct VableField {
    ident: Ident,
    role: VableRole,
    /// For `#[vable(static_field = N)]`: the VirtualizableInfo field index.
    static_field_index: Option<usize>,
}

fn parse_vable_role(s: &str) -> Option<VableRole> {
    match s {
        "frame" => Some(VableRole::Frame),
        "inputarg" => Some(VableRole::Inputarg),
        "info_only" => Some(VableRole::InfoOnly),
        "field" => Some(VableRole::Inputarg), // backward compat
        "array_base" => Some(VableRole::ArrayBase),
        "locals" => Some(VableRole::Locals),
        "stack" => Some(VableRole::Stack),
        "local_types" => Some(VableRole::LocalTypes),
        "stack_types" => Some(VableRole::StackTypes),
        "nlocals" => Some(VableRole::Nlocals),
        "valuestackdepth" => Some(VableRole::Valuestackdepth),
        _ => None,
    }
}

struct ParsedVableAttr {
    role: VableRole,
    static_field_index: Option<usize>,
}

/// Parse `#[vable(...)]` attribute content. Supports:
/// - Simple keyword: `#[vable(frame)]`, `#[vable(inputarg)]`
/// - Key-value: `#[vable(static_field = 0)]`
/// - Multi key-value: `#[vable(static_field = 0, type = ref)]`
fn parse_vable_attr(tokens_str: &str) -> Option<ParsedVableAttr> {
    let s = tokens_str.trim();
    // Check if it contains '=' (key-value pairs)
    if s.contains('=') {
        let mut static_idx = None;
        for part in s.split(',') {
            let part = part.trim();
            if let Some((key, val)) = part.split_once('=') {
                let key = key.trim();
                let val = val.trim();
                match key {
                    "static_field" => static_idx = val.parse().ok(),
                    "type" => {}
                    _ => {}
                }
            }
        }
        if static_idx.is_some() {
            return Some(ParsedVableAttr {
                role: VableRole::Frame, // overridden below
                static_field_index: static_idx,
            });
        }
        return None;
    }
    // Simple keyword (possibly with commas for combined: "inputarg, type = ref")
    if s.contains(',') {
        let parts: Vec<&str> = s.splitn(2, ',').collect();
        let keyword = parts[0].trim();
        let rest = parts.get(1).copied().unwrap_or("");
        if let Some(role) = parse_vable_role(keyword) {
            for part in rest.split(',') {
                let part = part.trim();
                if let Some((key, val)) = part.split_once('=') {
                    if key.trim() == "type" {
                        let _ = val.trim();
                    }
                }
            }
            return Some(ParsedVableAttr {
                role,
                static_field_index: None,
            });
        }
    }
    parse_vable_role(s).map(|role| ParsedVableAttr {
        role,
        static_field_index: None,
    })
}

fn extract_vable_fields(input: &DeriveInput) -> Vec<VableField> {
    let Data::Struct(data) = &input.data else {
        return Vec::new();
    };
    let Fields::Named(fields) = &data.fields else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for field in &fields.named {
        let Some(ident) = &field.ident else {
            continue;
        };
        for attr in &field.attrs {
            let Meta::List(meta_list) = &attr.meta else {
                continue;
            };
            if !meta_list.path.is_ident("vable") {
                continue;
            }
            let tokens_str = meta_list.tokens.to_string();
            if let Some(mut parsed) = parse_vable_attr(&tokens_str) {
                if parsed.static_field_index.is_some() {
                    parsed.role = VableRole::Inputarg; // state-backed
                }
                result.push(VableField {
                    ident: ident.clone(),
                    role: parsed.role,
                    static_field_index: parsed.static_field_index,
                });
            }
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════
// #[derive(VirtualizableSym)]
// ═══════════════════════════════════════════════════════════════

pub fn expand_sym(input: DeriveInput) -> TokenStream {
    let struct_name = &input.ident;
    let vable_fields = extract_vable_fields(&input);

    let frame_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::Frame)
        .map(|f| &f.ident);
    // Inputarg fields: included in jump/fail args and OpRef index assignment.
    let inputarg_fields: Vec<&Ident> = vable_fields
        .iter()
        .filter(|f| f.role == VableRole::Inputarg)
        .map(|f| &f.ident)
        .collect();
    // All static fields (inputarg + info_only): included in flush and oprefs.
    let all_static_fields: Vec<&Ident> = vable_fields
        .iter()
        .filter(|f| f.role == VableRole::Inputarg || f.role == VableRole::InfoOnly)
        .map(|f| &f.ident)
        .collect();
    let array_base_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::ArrayBase)
        .map(|f| &f.ident);
    let locals_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::Locals)
        .map(|f| &f.ident);
    // Stage 3.4 Phase C: `VableRole::Stack` is retired — the stack
    // now lives in the tail of `locals_field` (registers_r). The enum
    // variant is kept for attribute-parsing backward compatibility but
    // produces no extra binding here.
    let local_types_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::LocalTypes)
        .map(|f| &f.ident);
    let stack_types_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::StackTypes)
        .map(|f| &f.ident);
    let nlocals_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::Nlocals)
        .map(|f| &f.ident);
    let vsd_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::Valuestackdepth)
        .map(|f| &f.ident);

    // Generate flush: write const_int values to ALL static fields (heap layout order)
    let flush_writes: Vec<TokenStream> = all_static_fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            quote! { self.#field = ctx.const_int(values[#i]); }
        })
        .collect();
    let num_all_static = all_static_fields.len();

    // Generate field_values: read ALL static field OpRefs
    let field_value_reads: Vec<TokenStream> = all_static_fields
        .iter()
        .map(|field| {
            quote! { self.#field }
        })
        .collect();

    // stack_only_depth helper (if both vsd and nlocals fields exist)
    let stack_only_depth = if vsd_field.is_some() && nlocals_field.is_some() {
        let vsd = vsd_field.unwrap();
        let nl = nlocals_field.unwrap();
        quote! {
            /// Compute stack-only depth: valuestackdepth - nlocals.
            pub fn __vable_stack_only_depth(&self) -> usize {
                self.#vsd.saturating_sub(self.#nl)
            }
        }
    } else {
        quote! {}
    };

    // init_from_meta: assign OpRef indices to INPUTARG fields only.
    // The caller supplies the start because root inputargs are
    // [frame, extra_reds..., vable_scalars..., array...], while resume
    // virtualizable payloads are [vable, vable_scalars..., array...].
    let init_inputarg_fields: Vec<TokenStream> = inputarg_fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let offset = i as u32;
            quote! {
                self.#field = majit_ir::OpRef::from_raw(first_vable_scalar_idx + #offset);
            }
        })
        .collect();

    let init_frame = frame_field.map(|f| {
        quote! { self.#f = majit_ir::OpRef::input_arg_ref(0); }
    });

    let restore_inputarg_writes: Vec<TokenStream> = inputarg_fields
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let offset = i;
            quote! {
                self.#field = oprefs[first_vable_scalar_idx + #offset];
            }
        })
        .collect();
    let restore_inputarg_method = if frame_field.is_some() {
        let inputarg_count = inputarg_fields.len();
        quote! {
            /// Restore vable static-field OpRefs from an input slice.
            ///
            /// Use `FIRST_VABLE_SCALAR_IDX` for root inputargs
            /// (`[frame, extra_reds..., vable_scalars..., array...]`) and
            /// `1` for resume virtualizable payloads
            /// (`[vable, vable_scalars..., array...]`). The frame/vable
            /// identity in slot 0 is consumed by the caller and left
            /// untouched here, matching virtualizable.py:139-154.
            pub fn restore_inputarg_oprefs(
                &mut self,
                oprefs: &[majit_ir::OpRef],
                first_vable_scalar_idx: usize,
            ) -> usize {
                let __required = first_vable_scalar_idx + #inputarg_count;
                assert!(
                    oprefs.len() >= __required,
                    "restore_inputarg_oprefs: oprefs.len()={} < required={}",
                    oprefs.len(), __required,
                );
                #(#restore_inputarg_writes)*
                #inputarg_count
            }
        }
    } else {
        quote! {}
    };

    let init_array_base = array_base_field.map(|f| {
        let inputarg_count = inputarg_fields.len() as u32;
        quote! {
            self.#f = Some(first_vable_scalar_idx + #inputarg_count);
        }
    });

    // Collect methods
    let collect_frame = frame_field
        .map(|f| quote! { __args.push(self.#f); })
        .unwrap_or_default();
    let collect_inputargs: Vec<TokenStream> = inputarg_fields
        .iter()
        .map(|f| quote! { __args.push(self.#f); })
        .collect();

    let collect_typed_frame = frame_field
        .map(|f| quote! { __args.push((self.#f, majit_ir::Type::Ref)); })
        .unwrap_or_default();

    // Stage 3.4 Phase C: `locals_field` (registers_r) is the abstract
    // register file and carries both the locals [..nlocals] and the
    // stack tail [nlocals..nlocals+stack_only]. `collect_locals` emits
    // the locals window; `collect_stack` emits the stack window of the
    // same field. The RPython MIFrame equivalent is `registers_r` —
    // one contiguous vector indexed by abstract register color.
    let collect_locals = locals_field
        .map(|f| {
            if let Some(nl) = nlocals_field {
                quote! { __args.extend_from_slice(&self.#f[..self.#nl.min(self.#f.len())]); }
            } else {
                quote! { __args.extend_from_slice(&self.#f); }
            }
        })
        .unwrap_or_default();

    let collect_stack =
        if let (Some(lf), Some(_vsd), Some(nl)) = (locals_field, vsd_field, nlocals_field) {
            quote! {
                let __stack_only = self.__vable_stack_only_depth();
                let __nlocals = self.#nl;
                let __avail = self.#lf.len().saturating_sub(__nlocals);
                let __stack_len = __stack_only.min(__avail);
                __args.extend_from_slice(&self.#lf[__nlocals..__nlocals + __stack_len]);
            }
        } else {
            quote! {}
        };

    // Stage 3.4 Phase C: typed emission mirrors the locals+stack
    // window of `locals_field`. The `stack_types_field` side table
    // still provides per-stack-slot types (parallel to
    // `local_types_field` for locals); this layout matches RPython's
    // per-register type tagging (`history.py:262`) while keeping the
    // register storage unified.
    let collect_typed_locals = if let (Some(lf), Some(ltf)) = (locals_field, local_types_field) {
        if let Some(nl) = nlocals_field {
            quote! {
                let __locals_len = self.#nl.min(self.#lf.len());
                for (__i, &__opref) in self.#lf[..__locals_len].iter().enumerate() {
                    let __tp = self.#ltf.get(__i).copied().unwrap_or(majit_ir::Type::Ref);
                    __args.push((__opref, __tp));
                }
            }
        } else {
            quote! {
                for (__i, &__opref) in self.#lf.iter().enumerate() {
                    let __tp = self.#ltf.get(__i).copied().unwrap_or(majit_ir::Type::Ref);
                    __args.push((__opref, __tp));
                }
            }
        }
    } else {
        quote! {}
    };

    let collect_typed_stack = if let (Some(lf), Some(stf), Some(_vsd), Some(nl)) =
        (locals_field, stack_types_field, vsd_field, nlocals_field)
    {
        quote! {
            let __stack_only = self.__vable_stack_only_depth();
            let __nlocals = self.#nl;
            let __avail = self.#lf.len().saturating_sub(__nlocals);
            let __stack_len = __stack_only.min(__avail);
            for (__i, &__opref) in self.#lf[__nlocals..__nlocals + __stack_len].iter().enumerate() {
                let __tp = self.#stf.get(__i).copied().unwrap_or(majit_ir::Type::Ref);
                __args.push((__opref, __tp));
            }
        }
    } else {
        quote! {}
    };

    // Typed inputargs for collect_typed_jump_args (inputargs are always Int)
    let collect_typed_inputargs: Vec<TokenStream> = inputarg_fields
        .iter()
        .map(|f| {
            quote! { __args.push((self.#f, majit_ir::Type::Int)); }
        })
        .collect();

    quote! {
        impl #struct_name {
            /// Number of virtualizable static fields (excluding frame).
            pub const VABLE_NUM_STATIC_FIELDS: usize = #num_all_static;

            #stack_only_depth

            /// Flush virtualizable static fields from concrete values.
            ///
            /// `values` is `[last_instr, pycode, valuestackdepth, ...]`
            /// in VirtualizableInfo declared field order (interp_jit.py:25-31).
            pub fn flush_vable_fields(
                &mut self,
                ctx: &mut majit_metainterp::TraceCtx,
                values: &[i64],
            ) {
                debug_assert!(values.len() >= #num_all_static);
                #(#flush_writes)*
            }

            /// Read virtualizable static field OpRefs in declared order.
            pub fn vable_field_oprefs(&self) -> [majit_ir::OpRef; #num_all_static] {
                [#(#field_value_reads),*]
            }

            /// Initialize virtualizable OpRef indices from a layout offset.
            pub fn init_vable_indices(&mut self, first_vable_scalar_idx: u32) {
                #init_frame
                #(#init_inputarg_fields)*
                #init_array_base
            }

            #restore_inputarg_method

            /// Collect all virtualizable OpRefs in layout order for JUMP.
            pub fn vable_collect_jump_args(&self) -> Vec<majit_ir::OpRef> {
                let mut __args = Vec::new();
                #collect_frame
                #(#collect_inputargs)*
                #collect_locals
                #collect_stack
                __args
            }

            /// Collect all virtualizable typed OpRefs in layout order for JUMP.
            pub fn vable_collect_typed_jump_args(&self) -> Vec<(majit_ir::OpRef, majit_ir::Type)> {
                let mut __args = Vec::new();
                #collect_typed_frame
                #(#collect_typed_inputargs)*
                #collect_typed_locals
                #collect_typed_stack
                __args
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// #[derive(VirtualizableMeta)]
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VableMetaRole {
    NumLocals,
    Valuestackdepth,
    SlotTypes,
    HasVirtualizable,
}

fn parse_vable_meta_role(s: &str) -> Option<VableMetaRole> {
    match s {
        "num_locals" => Some(VableMetaRole::NumLocals),
        "valuestackdepth" => Some(VableMetaRole::Valuestackdepth),
        "slot_types" => Some(VableMetaRole::SlotTypes),
        "has_virtualizable" => Some(VableMetaRole::HasVirtualizable),
        _ => None,
    }
}

struct VableMetaField {
    ident: Ident,
    role: VableMetaRole,
}

fn extract_vable_meta_fields(input: &DeriveInput) -> Vec<VableMetaField> {
    let Data::Struct(data) = &input.data else {
        return Vec::new();
    };
    let Fields::Named(fields) = &data.fields else {
        return Vec::new();
    };

    let mut result = Vec::new();
    for field in &fields.named {
        let Some(ident) = &field.ident else {
            continue;
        };
        for attr in &field.attrs {
            let Meta::List(meta_list) = &attr.meta else {
                continue;
            };
            if !meta_list.path.is_ident("vable") {
                continue;
            }
            let role_str = meta_list.tokens.to_string();
            if let Some(role) = parse_vable_meta_role(role_str.trim()) {
                result.push(VableMetaField {
                    ident: ident.clone(),
                    role,
                });
            }
        }
    }
    result
}

pub fn expand_meta(input: DeriveInput) -> TokenStream {
    let struct_name = &input.ident;
    let vable_fields = extract_vable_meta_fields(&input);

    let num_locals_field = vable_fields
        .iter()
        .find(|f| f.role == VableMetaRole::NumLocals)
        .map(|f| &f.ident);
    let vsd_field = vable_fields
        .iter()
        .find(|f| f.role == VableMetaRole::Valuestackdepth)
        .map(|f| &f.ident);
    let slot_types_field = vable_fields
        .iter()
        .find(|f| f.role == VableMetaRole::SlotTypes)
        .map(|f| &f.ident);

    let stack_only_depth = if let (Some(vsd), Some(nl)) = (vsd_field, num_locals_field) {
        quote! {
            /// stack-only depth from meta: valuestackdepth - num_locals.
            pub fn vable_stack_only_depth(&self) -> usize {
                self.#vsd.saturating_sub(self.#nl)
            }
        }
    } else {
        quote! {}
    };

    // update_vsd_from_box_types: compute valuestackdepth from inputarg count
    let update_vsd = if let (Some(vsd), Some(st)) = (vsd_field, slot_types_field) {
        quote! {
            /// Update valuestackdepth and slot_types from inputarg/box type count.
            ///
            /// `box_len` is the total number of inputargs including scalars.
            /// `num_scalars` is the number of scalar inputargs (frame + fields).
            pub fn vable_update_vsd_from_len(&mut self, box_len: usize, num_scalars: usize) {
                if box_len >= num_scalars {
                    let new_vsd = box_len - num_scalars;
                    self.#vsd = new_vsd;
                    self.#st = vec![majit_ir::Type::Ref; new_vsd];
                }
            }
        }
    } else {
        quote! {}
    };

    quote! {
        impl #struct_name {
            #stack_only_depth
            #update_vsd
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// #[derive(VirtualizableState)]
// ═══════════════════════════════════════════════════════════════

pub fn expand_state(input: DeriveInput) -> TokenStream {
    let struct_name = &input.ident;
    let vable_fields = extract_vable_fields(&input);

    // Reject #[vable(static_field = N)] — heap is the single source of truth.
    // State-backed field mirroring was removed; all field access goes through
    // VirtualizableInfo on the heap. If static_field annotations are present,
    // emit a compile error instead of silently ignoring them.
    for f in &vable_fields {
        if f.static_field_index.is_some() {
            return syn::Error::new_spanned(
                &f.ident,
                "#[vable(static_field = N)] is no longer supported in VirtualizableState. \
                 Heap is the single source of truth; use heap accessors instead.",
            )
            .to_compile_error();
        }
    }

    let frame_field = vable_fields
        .iter()
        .find(|f| f.role == VableRole::Frame)
        .map(|f| &f.ident);

    let frame_ident = frame_field
        .cloned()
        .unwrap_or_else(|| format_ident!("frame"));

    // Pure VirtualizableInfo delegation (RPython parity).
    // Heap is the single source of truth — no state-backed fields.
    // All read/write goes through VirtualizableInfo which handles field types.
    quote! {
        impl #struct_name {
            /// virtualizable.py:86-99 read_boxes parity.
            /// Reads ALL static fields from the heap via VirtualizableInfo.
            pub fn virt_export_static_boxes(
                &self,
                info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> Vec<i64> {
                let heap_ptr = self.#frame_ident as *const u8;
                if !heap_ptr.is_null() {
                    unsafe { info.read_boxes(heap_ptr) }
                } else {
                    vec![0i64; info.num_fields()]
                }
            }

            /// virtualizable.py:126-137 write_from_resume_data_partial parity.
            /// Writes ALL static fields to the heap via VirtualizableInfo.
            pub fn virt_import_static_boxes(
                &mut self,
                info: &majit_metainterp::virtualizable::VirtualizableInfo,
                static_boxes: &[i64],
            ) -> bool {
                if static_boxes.len() < info.num_fields() {
                    return false;
                }
                let heap_ptr = self.#frame_ident as *mut u8;
                if heap_ptr.is_null() {
                    return false;
                }
                unsafe { info.write_boxes(heap_ptr, static_boxes); }
                true
            }

            /// virtualizable.py:86-99 read_boxes + array parity.
            /// Reads ALL static + array fields from heap via VirtualizableInfo.
            pub fn virt_export_all(
                &self,
                info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> (Vec<i64>, Vec<Vec<i64>>) {
                let heap_ptr = self.#frame_ident as *const u8;
                if heap_ptr.is_null() {
                    return (vec![0i64; info.num_fields()], vec![]);
                }
                let lengths = if info.can_read_all_array_lengths_from_heap() {
                    unsafe { info.read_array_lengths_from_heap(heap_ptr) }
                } else {
                    vec![]
                };
                unsafe { info.read_all_boxes(heap_ptr, &lengths) }
            }
        }
    }
}
