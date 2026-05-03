//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::quote;

use super::{JitInterpConfig, StateFieldKind};

/// Generate the JitState types and implementation.
pub fn generate_jit_state(config: &JitInterpConfig) -> TokenStream {
    generate_state_fields_jit_state(config)
}

/// Generate JitState types for state_fields mode (register/tape machines).
///
/// Instead of a storage pool with stacks, individual struct fields are tracked
/// as JIT-managed values. Scalars become single OpRefs, flattened arrays become
/// Vec<OpRef>, and virtualizable arrays (`[int; virt]`) track only a data
/// pointer + length OpRef pair (array stays on heap, accessed via raw memory ops).
fn generate_state_fields_jit_state(config: &JitInterpConfig) -> TokenStream {
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let sf = config.state_fields.as_ref().unwrap();

    let unsupported_fields: Vec<String> = sf
        .fields
        .iter()
        .filter_map(|f| match &f.kind {
            StateFieldKind::Scalar { ir_type, .. } => {
                let ty = ir_type.to_string();
                if ty == "int" {
                    None
                } else {
                    Some(format!("{}: {}", f.name, ty))
                }
            }
            StateFieldKind::Array(tp) | StateFieldKind::VirtArray(tp) => {
                let ty = tp.to_string();
                if ty == "int" {
                    None
                } else {
                    Some(format!("{}: {}", f.name, ty))
                }
            }
            // RPython parity: opaque(T) fields are pass-through; the JIT
            // does not enumerate them as inputargs, so any T is allowed.
            StateFieldKind::Opaque(_) => None,
        })
        .collect();
    if !unsupported_fields.is_empty() {
        let message = format!(
            "state_fields supports int, [int], [int; virt], and opaque(T); unsupported: {}",
            unsupported_fields.join(", ")
        );
        return quote! {
            compile_error!(#message);
        };
    }

    // Separate scalars, flattened arrays, and virtualizable arrays.
    let scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Scalar { .. }))
        .collect();
    // Helper: per-scalar Rust storage type token (`i64` by default, or
    // the explicit `int(<TypePath>)` override). Used to emit `as <type>`
    // casts at the JIT boundary so user struct fields can stay in their
    // natural Rust types (e.g. `selected: usize`, `stacksize: i32`).
    let scalar_rust_type = |kind: &StateFieldKind| -> TokenStream {
        match kind {
            StateFieldKind::Scalar {
                rust_type: Some(p), ..
            } => quote! { #p },
            _ => quote! { i64 },
        }
    };
    let arrays: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Array(_)))
        .collect();
    let virt_arrays: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::VirtArray(_)))
        .collect();

    let num_scalars = scalars.len();
    let num_virt_arrays = virt_arrays.len();

    // ── __JitMeta fields: one `{name}_len: usize` per flattened array ──
    // Virt arrays do NOT store length in meta (it's dynamic, tracked as inputarg).
    let meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: usize, }
        })
        .collect();

    // ── __JitSym fields ──
    // scalar → OpRef
    // flattened array → Vec<OpRef>
    // virt array → (OpRef, OpRef) for (data_ptr, len)
    let sym_scalar_fields: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: majit_ir::OpRef, }
        })
        .collect();
    let sym_scalar_value_fields: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #value_name: i64, }
        })
        .collect();
    let sym_array_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: Vec<majit_ir::OpRef>, }
        })
        .collect();
    let sym_array_value_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! { #value_name: Vec<i64>, }
        })
        .collect();
    let sym_virt_array_fields: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                #ptr_name: majit_ir::OpRef,
                #len_name: majit_ir::OpRef,
                #ptr_value_name: i64,
                #len_value_name: i64,
            }
        })
        .collect();

    // ── JitCodeSym: total_slots ──
    // num_scalars + sum(flattened array lengths) + 2 * num_virt_arrays
    let total_slots_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { + self.#fname.len() }
        })
        .collect();

    // ── JitCodeSym: state_field_ref / set_state_field_ref ──
    let state_field_ref_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let idx_lit = idx;
            // OpRef field (Sym side) — return as-is.
            quote! { #idx_lit => Some(self.#fname), }
        })
        .collect();
    let set_state_field_ref_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let idx_lit = idx;
            // OpRef field on Sym — direct assignment, no cast.
            quote! { #idx_lit => { self.#fname = value; } }
        })
        .collect();
    let state_field_value_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            let idx_lit = idx;
            quote! { #idx_lit => Some(self.#value_name), }
        })
        .collect();
    let set_state_field_value_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            let idx_lit = idx;
            quote! { #idx_lit => { self.#value_name = value; } }
        })
        .collect();

    // ── JitCodeSym: state_array_ref / set_state_array_ref (flattened only) ──
    let state_array_ref_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let fname = &f.name;
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => self.#fname.get(elem_idx).copied(), }
        })
        .collect();
    let set_state_array_ref_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let fname = &f.name;
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => {
                if elem_idx < self.#fname.len() {
                    self.#fname[elem_idx] = value;
                }
            } }
        })
        .collect();
    let state_array_value_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_values", f.name);
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => self.#value_name.get(elem_idx).copied(), }
        })
        .collect();
    let set_state_array_value_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_values", f.name);
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => {
                if elem_idx < self.#value_name.len() {
                    self.#value_name[elem_idx] = value;
                }
            } }
        })
        .collect();

    // ── JitCodeSym: state_varray_ptr / state_varray_len ──
    let state_varray_ptr_arms: Vec<TokenStream> = virt_arrays
        .iter()
        .enumerate()
        .map(|(va_idx, (_, f))| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let va_idx_lit = va_idx;
            quote! { #va_idx_lit => Some(self.#ptr_name), }
        })
        .collect();
    let state_varray_len_arms: Vec<TokenStream> = virt_arrays
        .iter()
        .enumerate()
        .map(|(va_idx, (_, f))| {
            let len_name = quote::format_ident!("{}_len", f.name);
            let va_idx_lit = va_idx;
            quote! { #va_idx_lit => Some(self.#len_name), }
        })
        .collect();

    // ── collect_jump_args: scalars, then flattened arrays, then virt array ptr+len ──
    let collect_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(sym.#fname); }
        })
        .collect();
    let collect_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.extend_from_slice(&sym.#fname); }
        })
        .collect();
    let collect_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                args.push(sym.#ptr_name);
                args.push(sym.#len_name);
            }
        })
        .collect();

    // ── populate_frame_int_regs: scalars + flattened arrays ──
    // Matches `live_slots_for_state_field_jit` slot order so a
    // `MIFrame::get_list_of_active_boxes` walk against the canonical
    // liveness entry decodes back the same OpRefs / values that
    // `__JitSym` and the macro-emitted `live/<offset>` placeholder
    // refer to.  Virt-array populate is deferred (Session 3b) — see
    // the trait-method docstring.
    let populate_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(self.#fname);
                    frame.int_values[__slot] = Some(self.#value_name);
                }
                __slot += 1;
            }
        })
        .collect();
    let populate_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                for __i in 0..self.#fname.len() {
                    if __slot + __i < frame.int_regs.len() {
                        frame.int_regs[__slot + __i] = Some(self.#fname[__i]);
                        frame.int_values[__slot + __i] = Some(self.#value_name[__i]);
                    }
                }
                __slot += self.#fname.len();
            }
        })
        .collect();
    // Virt-array populate (Session 3b-1): two consecutive slots per
    // varray — `<varr>_ptr` (data pointer OpRef) at offset N, then
    // `<varr>_len` at N+1.  Value mirrors come from
    // `<varr>_ptr_value` / `<varr>_len_value` cached at
    // `JitState::initialize_sym` from the user state's
    // `<varr>.as_ptr() as i64` / `<varr>.len() as i64`.
    let populate_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(self.#ptr_name);
                    frame.int_values[__slot] = Some(self.#ptr_value_name);
                }
                __slot += 1;
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(self.#len_name);
                    frame.int_values[__slot] = Some(self.#len_value_name);
                }
                __slot += 1;
            }
        })
        .collect();

    // ── fail_args ──
    let fail_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(self.#fname); }
        })
        .collect();
    let fail_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.extend_from_slice(&self.#fname); }
        })
        .collect();
    let fail_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                args.push(self.#ptr_name);
                args.push(self.#len_name);
            }
        })
        .collect();

    // ── build_meta: capture flattened array lengths ──
    let build_meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: self.#fname.len(), }
        })
        .collect();

    // ── canonical_liveness_slots: array_lens slice expression ──
    // RPython `assembler.py:218-231 get_liveness_info` extracts per-kind
    // liveness for each `-live-` marker.  In flat-state JIT every slot
    // is permanently live, so the canonical entry is just
    // `[0..total_slots]` of int slots.  The `array_lens` slice fed to
    // `live_slots_for_state_field_jit` enumerates the runtime lengths
    // captured in `__JitMeta::<arr>_len` (one per flattened array).
    let canonical_liveness_array_len_refs: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { self.#len_name }
        })
        .collect();

    // ── extract_live: scalars, then flattened array elements, then virt array ptr+len ──
    let extract_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { values.push(self.#fname as i64); }
        })
        .collect();
    let extract_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for elem in &self.#fname {
                    values.push(*elem as i64);
                }
            }
        })
        .collect();
    let extract_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                values.push(self.#fname.as_ptr() as i64);
                values.push(self.#fname.len() as i64);
            }
        })
        .collect();

    // ── create_sym: assign sequential OpRef::from_raw(0), OpRef::from_raw(1), ... ──
    let create_sym_scalar_inits: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                let #fname = majit_ir::OpRef::input_arg_int(__offset as u32);
                __offset += 1;
                let #value_name = 0i64;
            }
        })
        .collect();
    let create_sym_array_inits: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                let #fname: Vec<majit_ir::OpRef> = (0..meta.#len_name)
                    .map(|i| {
                        let r = majit_ir::OpRef::from_raw((__offset + i) as u32);
                        r
                    })
                    .collect();
                let #value_name: Vec<i64> = vec![0; meta.#len_name];
                __offset += meta.#len_name;
            }
        })
        .collect();
    let create_sym_virt_array_inits: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                let #ptr_name = majit_ir::OpRef::input_arg_ref(__offset as u32);
                __offset += 1;
                let #len_name = majit_ir::OpRef::input_arg_int(__offset as u32);
                __offset += 1;
                let #ptr_value_name = 0i64;
                let #len_value_name = 0i64;
            }
        })
        .collect();
    let create_sym_scalar_names: Vec<&syn::Ident> = scalars.iter().map(|(_, f)| &f.name).collect();
    let create_sym_array_names: Vec<&syn::Ident> = arrays.iter().map(|(_, f)| &f.name).collect();
    let create_sym_scalar_value_names: Vec<syn::Ident> = scalars
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_value", f.name))
        .collect();
    let create_sym_array_value_names: Vec<syn::Ident> = arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_values", f.name))
        .collect();
    let create_sym_virt_array_ptr_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_ptr", f.name))
        .collect();
    let create_sym_virt_array_len_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len", f.name))
        .collect();
    let create_sym_virt_array_ptr_value_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_ptr_value", f.name))
        .collect();
    let create_sym_virt_array_len_value_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len_value", f.name))
        .collect();

    // ── is_compatible: check flattened array lengths match meta ──
    // Virt arrays always compatible (ptr+len are inputargs, not fixed).
    let compat_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && self.#fname.len() == meta.#len_name }
        })
        .collect();

    // ── restore: write values back to state fields ──
    // Virt arrays: restore ptr (ignored, Vec owns it) and skip len.
    // The compiled code writes directly to the heap backing the Vec, so
    // no element-level restore is needed.
    let restore_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                self.#fname = values[__offset] as #rust_ty;
                __offset += 1;
            }
        })
        .collect();
    let restore_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                let __arr_len = self.#fname.len();
                for i in 0..__arr_len {
                    self.#fname[i] = values[__offset + i];
                }
                __offset += __arr_len;
            }
        })
        .collect();
    let restore_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|_| {
            // Skip the 2 slots (ptr + len) — virt array data lives on heap,
            // already modified in-place by compiled code.
            quote! {
                __offset += 2;
            }
        })
        .collect();
    let initialize_sym_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            // Cast user's typed field to i64 for the JIT Sym slot.
            quote! {
                sym.#value_name = self.#fname as i64;
            }
        })
        .collect();
    let initialize_sym_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                sym.#value_name.clone_from(&self.#fname);
            }
        })
        .collect();
    // Virt-array `<varr>_ptr_value` / `<varr>_len_value` mirror the
    // current `<state>.<varr>` ptr/len so `populate_frame_int_regs`
    // can fill the corresponding `MIFrame.int_values` slots without
    // re-reading the live state at guard time
    // (Task #89 framestack-lift Session 3b-1).  PRE-EXISTING-ADAPTATION:
    // accurate iff the user's varray Vec does not reallocate during
    // tracing — true for the 6 macro examples
    // (`vec![0i64; program.len()]` is fixed-capacity).  Dynamic
    // varrays would need per-mutation refresh hooks.
    let initialize_sym_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                sym.#ptr_value_name = self.#fname.as_ptr() as i64;
                sym.#len_value_name = self.#fname.len() as i64;
            }
        })
        .collect();

    // ── validate_close: flattened array lengths in sym match meta ──
    // Virt arrays always validate (ptr+len are just OpRefs, not sized).
    let validate_array_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && sym.#fname.len() == meta.#len_name }
        })
        .collect();

    quote! {
        /// Compiled loop metadata for state_fields mode: flattened array lengths at trace start.
        #[derive(Clone)]
        #[allow(non_camel_case_types)]
        struct __JitMeta {
            #(#meta_fields)*
        }

        impl __JitMeta {
            /// RPython `assembler.py:218-231 get_liveness_info(insn, kind)`
            /// adapted for flat-state JIT: every state_field slot is
            /// permanently live, so the canonical
            /// `(live_i, live_r, live_f)` triple is just `0..total_slots`
            /// in the int bank.  state-field JIT enforces `Type::Int` on
            /// every slot at macro expansion (`codegen_state.rs:30-43`),
            /// so `live_r` and `live_f` are empty.  Used by
            /// `JitCodeBuilder::live` (`assembler.py:148+158`) to
            /// register the canonical entry once per process and emit
            /// a `live/<offset>` prefix on each per-opcode jitcode.
            #[allow(dead_code)]
            fn canonical_liveness_slots(
                &self,
            ) -> (::std::vec::Vec<u8>, ::std::vec::Vec<u8>, ::std::vec::Vec<u8>) {
                let __array_lens: &[usize] = &[#(#canonical_liveness_array_len_refs),*];
                majit_metainterp::live_slots_for_state_field_jit(
                    #num_scalars,
                    __array_lens,
                    #num_virt_arrays,
                )
            }

            /// RPython `warmspot.py:281-289`'s `make_jitcodes() →
            /// finish_setup(codewriter)` lifecycle reduced to the
            /// canonical-entry slice for state-field JIT
            /// (`pyjitpl.py:2264 self.liveness_info = "".join(
            /// asm.all_liveness)`).  Builds a fresh `Assembler`,
            /// registers the canonical
            /// `(live_i, live_r, live_f)` triple via
            /// `Assembler::_encode_liveness` (`assembler.py:235-248`),
            /// then publishes the resulting `all_liveness` payload
            /// through `JitDriver::install_canonical_liveness`.
            ///
            /// Caller pattern:
            /// ```ignore
            /// let meta = state.build_meta(0, &program);
            /// meta.install_canonical_liveness(&mut driver);
            /// ```
            /// Must run before the first trace — the
            /// `Arc::get_mut` invariant on `MetaInterp::staticdata`
            /// (`pyjitpl/mod.rs::install_canonical_liveness`) panics
            /// once any tracing setup has cloned the Arc.
            ///
            /// KNOWN GAP vs RPython `assembler.py:146-158 write_insn`:
            /// RPython emits a per-guard `-live-` marker whose
            /// `(live_i, live_r, live_f)` is computed from the next
            /// instruction's `Register` operands (`get_liveness_info`),
            /// dedupes the entry into `all_liveness_positions`, and
            /// patches each BC_LIVE marker's 2-byte offset to point at
            /// that specific entry (`encode_offset`).  pyre's macro-
            /// emitted JitCodes call `JitCodeBuilder::live_placeholder`
            /// which writes BC_LIVE + zero offset, so every guard's
            /// blackhole resume reads back this single canonical
            /// entry — an over-approximation that claims all SYM
            /// scalars / arrays / virt_arrays are live regardless of
            /// the per-pc liveness actually demanded by the next op.
            /// For aheui (today's sole `#[jit_interp]` user) the per-
            /// opcode JitCode body touches every SYM slot, so the
            /// canonical answer is also precise; the gap is purely
            /// structural for future polymorphic state-field JIT
            /// consumers.  Closing it requires the lowerer to track
            /// per-pc live slot sets during JitCode emission and
            /// register one liveness entry per distinct set, then
            /// patch each `live_placeholder` via
            /// `JitCodeBuilder::patch_live_offset` to point at the
            /// matching entry — orthogonal to this canonical-bridge
            /// shim.
            #[allow(dead_code)]
            fn install_canonical_liveness(
                &self,
                driver: &mut majit_metainterp::JitDriver<#state_type>,
            ) {
                let mut __asm = majit_metainterp::Assembler::new();
                let (__live_i, __live_r, __live_f) = self.canonical_liveness_slots();
                let mut __code_buf: ::std::vec::Vec<u8> = ::std::vec::Vec::new();
                __asm._encode_liveness(
                    &__live_i,
                    &__live_r,
                    &__live_f,
                    &mut __code_buf,
                );
                // RPython `assembler.py:222 self.insns[key] = opnum`
                // records every opcode the assembler emits during
                // `assemble()`.  pyre's macro path skips
                // `assembler.assemble()` (the per-arm `JitCodeBuilder`
                // emits BC_* directly), so the canonical state-field
                // JIT entries are registered explicitly here.  The
                // downstream `MetaInterpStaticData::
                // install_canonical_liveness` then calls
                // `setup_insns(asm.insns())` (`pyjitpl.py:2227-2243`)
                // to dynamically resolve `op_live` /
                // `op_catch_exception` / `op_*_return` instead of a
                // parallel hardcoded `BC_*` seeding block.
                __asm.register_insn("live/", majit_metainterp::BC_LIVE);
                __asm.register_insn(
                    "catch_exception/L",
                    majit_metainterp::BC_CATCH_EXCEPTION,
                );
                __asm.register_insn(
                    "rvmprof_code/ii",
                    majit_metainterp::BC_RVMPROF_CODE,
                );
                __asm.register_insn("int_return/i", majit_metainterp::BC_INT_RETURN);
                __asm.register_insn("ref_return/r", majit_metainterp::BC_REF_RETURN);
                __asm.register_insn(
                    "float_return/f",
                    majit_metainterp::BC_FLOAT_RETURN,
                );
                __asm.register_insn("void_return/", majit_metainterp::BC_VOID_RETURN);
                driver.install_canonical_liveness(&__asm);
            }
        }

        /// Symbolic state during tracing: per-field OpRefs.
        #[allow(non_camel_case_types)]
        struct __JitSym {
            #(#sym_scalar_fields)*
            #(#sym_scalar_value_fields)*
            #(#sym_array_fields)*
            #(#sym_array_value_fields)*
            #(#sym_virt_array_fields)*
            loop_header_pc: usize,
            current_portal_pc: usize,
            trace_started: bool,
        }

        impl majit_metainterp::JitCodeSym for __JitSym {
            fn total_slots(&self) -> usize {
                #num_scalars #(#total_slots_array_parts)* + #num_virt_arrays * 2
            }

            fn loop_header_pc(&self) -> usize {
                self.loop_header_pc
            }

            fn begin_portal_op(&mut self, pc: usize) {
                self.current_portal_pc = pc;
            }

            fn state_field_ref(&self, field_idx: usize) -> Option<majit_ir::OpRef> {
                match field_idx {
                    #(#state_field_ref_arms)*
                    _ => None,
                }
            }

            fn set_state_field_ref(&mut self, field_idx: usize, value: majit_ir::OpRef) {
                match field_idx {
                    #(#set_state_field_ref_arms)*
                    _ => {}
                }
            }

            fn state_field_value(&self, field_idx: usize) -> Option<i64> {
                match field_idx {
                    #(#state_field_value_arms)*
                    _ => None,
                }
            }

            fn set_state_field_value(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#set_state_field_value_arms)*
                    _ => {}
                }
            }

            fn state_array_ref(&self, array_idx: usize, elem_idx: usize) -> Option<majit_ir::OpRef> {
                match array_idx {
                    #(#state_array_ref_arms)*
                    _ => None,
                }
            }

            fn set_state_array_ref(&mut self, array_idx: usize, elem_idx: usize, value: majit_ir::OpRef) {
                match array_idx {
                    #(#set_state_array_ref_arms)*
                    _ => {}
                }
            }

            fn state_array_value(&self, array_idx: usize, elem_idx: usize) -> Option<i64> {
                match array_idx {
                    #(#state_array_value_arms)*
                    _ => None,
                }
            }

            fn set_state_array_value(&mut self, array_idx: usize, elem_idx: usize, value: i64) {
                match array_idx {
                    #(#set_state_array_value_arms)*
                    _ => {}
                }
            }

            fn state_varray_ptr(&self, array_idx: usize) -> Option<majit_ir::OpRef> {
                match array_idx {
                    #(#state_varray_ptr_arms)*
                    _ => None,
                }
            }

            fn state_varray_len(&self, array_idx: usize) -> Option<majit_ir::OpRef> {
                match array_idx {
                    #(#state_varray_len_arms)*
                    _ => None,
                }
            }

            fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                let mut args = Vec::new();
                #(#fail_scalar_parts)*
                #(#fail_array_parts)*
                #(#fail_virt_array_parts)*
                Some(args)
            }

            #[allow(unused_assignments, unused_variables)]
            fn populate_frame_int_regs(
                &self,
                frame: &mut majit_metainterp::MIFrame,
            ) {
                // Slot layout matches `live_slots_for_state_field_jit`
                // (Task #89 orth-6): scalars at `0..num_scalars`,
                // then flattened arrays, then virt-array (ptr, len)
                // pairs.  Virt-array value mirrors are cached at
                // `JitState::initialize_sym` time
                // (Task #89 framestack-lift Session 3b-1) from the
                // user state's `<varr>.as_ptr()` / `<varr>.len()`,
                // accurate iff the Vec does not reallocate during
                // tracing.
                let mut __slot: usize = 0;
                #(#populate_scalar_parts)*
                #(#populate_array_parts)*
                #(#populate_virt_array_parts)*
                let _ = __slot;
            }
        }

        impl majit_metainterp::JitState for #state_type {
            type Meta = __JitMeta;
            type Sym = __JitSym;
            type Env = #env_type;

            fn can_trace(&self) -> bool {
                true
            }

            fn build_meta(&self, _header_pc: usize, _program: &#env_type) -> __JitMeta {
                __JitMeta {
                    #(#build_meta_fields)*
                }
            }

            fn extract_live(&self, _meta: &__JitMeta) -> Vec<i64> {
                let mut values = Vec::new();
                #(#extract_scalar_parts)*
                #(#extract_array_parts)*
                #(#extract_virt_array_parts)*
                values
            }

            fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                let mut __offset: usize = 0;
                #(#create_sym_scalar_inits)*
                #(#create_sym_array_inits)*
                #(#create_sym_virt_array_inits)*
                __JitSym {
                    #(#create_sym_scalar_names,)*
                    #(#create_sym_scalar_value_names,)*
                    #(#create_sym_array_names,)*
                    #(#create_sym_array_value_names,)*
                    #(#create_sym_virt_array_ptr_names,)*
                    #(#create_sym_virt_array_len_names,)*
                    #(#create_sym_virt_array_ptr_value_names,)*
                    #(#create_sym_virt_array_len_value_names,)*
                    loop_header_pc: header_pc,
                    current_portal_pc: header_pc,
                    trace_started: false,
                }
            }

            fn initialize_sym(&self, sym: &mut __JitSym, _meta: &__JitMeta) {
                #(#initialize_sym_scalar_parts)*
                #(#initialize_sym_array_parts)*
                #(#initialize_sym_virt_array_parts)*
            }

            fn is_compatible(&self, meta: &__JitMeta) -> bool {
                true #(#compat_checks)*
            }

            fn restore(&mut self, _meta: &__JitMeta, values: &[i64]) {
                let mut __offset: usize = 0;
                #(#restore_scalar_parts)*
                #(#restore_array_parts)*
                #(#restore_virt_array_parts)*
            }

            fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                #(#collect_scalar_parts)*
                #(#collect_array_parts)*
                #(#collect_virt_array_parts)*
                args
            }

            fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                true #(#validate_array_checks)*
            }

            // Task #89 framestack-lift Session 4: state-field JIT
            // override of `JitState::populate_frame_for_guard` so
            // jitdriver-level guard sites (e.g. `force_finish_trace`'s
            // GuardAlwaysFails fallback) get the same snapshot wire-up
            // as the dispatch-level `record_state_guard`
            // (`pyjitpl/dispatch.rs:284`).  Calls the macro-emitted
            // `JitCodeSym::populate_frame_int_regs` to bridge
            // `__JitSym` slots onto `MIFrame.int_regs`, then builds a
            // single-frame snapshot via the canonical helper.
            fn populate_frame_for_guard(
                sym: &__JitSym,
                frames: &mut majit_metainterp::MIFrameStack,
                __op_live: u8,
                __all_liveness: &[u8],
            ) -> Option<majit_metainterp::recorder::Snapshot> {
                use majit_metainterp::JitCodeSym as _;
                if frames.frames.is_empty() {
                    return None;
                }
                let __total_slots = sym.total_slots();
                let __root = &mut frames.frames[0];
                let __n = __total_slots.min(__root.int_regs.len());
                let __saved_int_regs: Vec<Option<majit_ir::OpRef>> =
                    __root.int_regs[..__n].to_vec();
                let __saved_int_values: Vec<Option<i64>> =
                    __root.int_values[..__n].to_vec();
                sym.populate_frame_int_regs(__root);
                // Fallback path (jitdriver-level GuardAlwaysFails): no
                // resume_pc swap has happened, so use the stable portal
                // pc captured by `begin_portal_op`, in the same factory-key
                // convention as dispatch-level state guards.
                let program_pc = (sym.current_portal_pc + 1) as u32;
                let mut __pool = majit_metainterp::ConstantPool::new();
                let __snapshot = majit_metainterp::build_state_field_snapshot(
                    frames,
                    program_pc,
                    __op_live,
                    __all_liveness,
                    &mut __pool,
                );
                let __root = &mut frames.frames[0];
                __root.int_regs[..__n].copy_from_slice(&__saved_int_regs);
                __root.int_values[..__n].copy_from_slice(&__saved_int_values);
                Some(__snapshot)
            }
        }
    }
}
