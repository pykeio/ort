use heck::ToUpperCamelCase;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{
	parse::{Parse, ParseStream},
	parse_macro_input,
	punctuated::Punctuated,
	spanned::Spanned,
	BinOp, Expr, ExprAssign, ItemFn, Lit, LitStr, Path, ReturnType, Token, Type
};

#[derive(Default)]
struct OperatorMeta {
	name: Option<String>,
	krate: Option<Path>
}

impl Parse for OperatorMeta {
	fn parse(input: ParseStream) -> syn::Result<Self> {
		let mut opts = Punctuated::<TokenStream, Token![,]>::parse_terminated(input)?;
		let mut meta = OperatorMeta::default();
		while let Some(pair) = opts.pop() {
			let inner = pair.into_value();
			if let Ok(display_name) = syn::parse2::<LitStr>(inner.clone()) {
				meta.name = Some(display_name.value());
				continue;
			}

			let span = inner.span();
			match syn::parse2::<ExprAssign>(inner.clone()) {
				Ok(expr) => {
					let Expr::Path(property_path) = &*expr.left else {
						return Err(syn::Error::new(expr.left.span(), "left-hand side of configuration option should be a simple identifier"));
					};
					let property_ident = property_path
						.path
						.require_ident()
						.map_err(|e| syn::Error::new(e.span(), "left-hand side of configuration option should be a simple identifier"))?;

					match property_ident.to_string().as_str() {
						"name" => {
							let Expr::Lit(lit) = &*expr.right else {
								return Err(syn::Error::new(expr.right.span(), "`name` option takes a string"));
							};
							let Lit::Str(name) = &lit.lit else {
								return Err(syn::Error::new(lit.span(), "`name` option takes a string"));
							};
							meta.name = Some(name.value());
						}
						"ort" => {
							let Expr::Path(path) = &*expr.right else {
								return Err(syn::Error::new(expr.right.span(), "`ort` option expects a path"));
							};
							meta.krate = Some(path.path.to_owned());
						}
						x => return Err(syn::Error::new(property_ident.span(), format!("unknown configuration option `{x}`")))
					}
				}
				Err(_) => return Err(syn::Error::new(span, "expected a configuration option like `a = b`"))
			}
		}
		Ok(meta)
	}
}

enum KernelAttributeType {
	Float,
	Int64,
	String,
	Tensor,
	FloatArray,
	Int64Array,
	ConstantTensor
}

struct KernelAttribute {
	r#type: KernelAttributeType,
	name: String,
	optional: bool
}

enum KernelInput {
	Tensor(Type),
	OptionalTensor(Type),
	VariadicTensor(Option<Type>)
}

enum KernelOutput {
	Tensor(Type),
	OptionalTensor(Type),
	VariadicTensor(Option<Type>)
}

struct KernelParameters {
	inputs: Vec<KernelInput>,
	outputs: Vec<KernelOutput>,
	attributes: Vec<KernelAttribute>
}

#[proc_macro_attribute]
pub fn operator(attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
	let item: ItemFn = parse_macro_input!(item as ItemFn);
	let meta = parse_macro_input!(attr as OperatorMeta);
	let operator_name = meta.name.unwrap_or_else(|| item.sig.ident.to_string());
	let ort = meta.krate.unwrap_or_else(|| syn::parse_str("::ort").unwrap());

	let kernel_ident = Ident::new(&format!("{}Kernel", operator_name.to_upper_camel_case()), Span::call_site());
	let operator_ident = Ident::new(&operator_name, Span::call_site());
	let vis = &item.vis;

	let body = item.block;
	let (ret_value, ret_type) = match item.sig.output {
		ReturnType::Default => (Some(quote!(Ok(()))), Box::new(syn::parse_quote!(#ort::Result<()>))),
		ReturnType::Type(_, ty) => (None, ty)
	};

	let operator_name_lit = LitStr::new(&operator_name, Span::call_site());

	proc_macro::TokenStream::from(quote! {
		#[doc(hidden)]
		struct #kernel_ident;

		impl #ort::Kernel for #kernel_ident {
			fn compute(&mut self, ctx: &mut #ort::KernelContext) -> #ret_type {
				#body
				#ret_value
			}
		}

		#[allow(non_snake_case)]
		#vis struct #operator_ident;

		impl #ort::Operator for #operator_ident {
			type Kernel = #kernel_ident;

			fn name() -> &'static str {
				#operator_name_lit
			}

			fn create_kernel(_: #ort::KernelAttributes) -> #ort::Result<Self::Kernel> {
				Ok(#kernel_ident)
			}

			fn inputs() -> Vec<#ort::OperatorInput> {
				vec![]
			}

			fn outputs() -> Vec<#ort::OperatorOutput> {
				vec![]
			}
		}
	})
}
