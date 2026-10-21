// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// GGUF reader, pure Rust (no llama.cpp): the file layout of ggml's gguf.cpp (GGUF v2/v3) — header, metadata KVs of
// every value type (arrays included), tensor infos, the alignment (general.alignment, default 32) and the data section.
// The model is NEVER held in CPU memory (the browser build would run out of memory on a 1.5 GB copy):
// the header is parsed from a prefix of the file (1 MB, doubled while the header is longer — the text GGUF's tokenizer
// makes it ≈ 6 MB), and a tensor is only (file offset, byte length, ggml type, shape). Its bytes come from a
// TensorSource ("read range [off, off+len) into this buffer": a native file here, an OPFS sync access handle in the
// browser later) straight into the GPU upload's staging buffer (gpu.rs).
// Only the types the OCR model uses are readable (F32, F16, BF16, Q8_0); any other type is an error, never converted.

use std::collections::BTreeMap;
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const FIRST_PREFIX: usize = 1 << 20;

/// random-access reads of the model file: fill `buf` with the bytes at [off, off + buf.len())
pub trait TensorSource {
    fn len(&self) -> Result<u64>;
    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()>;
}

/// a native file (pread; no mmap — mapped pages would count against the process like a copy)
#[cfg(not(target_arch = "wasm32"))]
pub struct FileSource(File);

#[cfg(not(target_arch = "wasm32"))]
impl FileSource {
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self(File::open(path).with_context(|| format!("cannot open {}", path.display()))?))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl TensorSource for FileSource {
    fn len(&self) -> Result<u64> {
        Ok(self.0.metadata()?.len())
    }

    #[cfg(unix)]
    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        self.0.read_exact_at(buf, off).with_context(|| format!("read {} bytes at {off}", buf.len()))
    }

    #[cfg(windows)]
    fn read_at(&self, mut off: u64, mut buf: &mut [u8]) -> Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = self.0.seek_read(buf, off)?;
            if n == 0 {
                bail!("file ends at {off}");
            }
            buf = &mut buf[n..];
            off += n as u64;
        }
        Ok(())
    }
}

/// bytes in memory (tests)
impl TensorSource for Vec<u8> {
    fn len(&self) -> Result<u64> {
        Ok(self.as_slice().len() as u64)
    }

    fn read_at(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        let end = off as usize + buf.len();
        if end > self.as_slice().len() {
            bail!("read past the end ({end} > {})", self.as_slice().len());
        }
        buf.copy_from_slice(&self[off as usize..end]);
        Ok(())
    }
}

/// the header did not fit in the prefix read so far
#[derive(Debug)]
struct NeedMore;

impl fmt::Display for NeedMore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGUF header longer than the prefix read")
    }
}

impl std::error::Error for NeedMore {}

/// ggml tensor types this engine reads (ggml.h enum ggml_type values)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    BF16,
    Q8_0,
}

impl TensorType {
    fn from_ggml(t: u32) -> Result<Self> {
        match t {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            30 => Ok(Self::BF16),
            _ => bail!("ggml type {t} is not supported (F32, F16, BF16, Q8_0 only)"),
        }
    }

    /// the ggml name (q8_0, bf16, …)
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::Q8_0 => "q8_0",
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34), // f16 scale + 32 × i8
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<Value>),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v:?}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
            Self::Array(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().take(8).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                if a.len() > 8 {
                    write!(f, ", … ({} items)", a.len())?;
                }
                write!(f, "]")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: shape[0] is the contiguous (innermost) dimension
    pub shape: Vec<u64>,
    pub ty: TensorType,
    /// offset from the start of the data section
    pub offset: u64,
    /// absolute offset in the file
    pub file_offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn n_bytes(&self) -> u64 {
        let (be, bb) = self.ty.block();
        self.n_elements() / be * bb
    }
}

/// the status / settings line of a model pair: the formats the weights are STORED in ("q8_0 text · bf16 vision") —
/// not the compute format (the wgpu engine's Gpu::compute_label)
pub fn weights_label<T: TensorSource, M: TensorSource>(text: &Gguf<T>, mmproj: &Gguf<M>) -> Result<String> {
    Ok(format!("{} text · {} vision", text.matrix_type()?.name(), mmproj.matrix_type()?.name()))
}

/// the parsed header + the source the tensor bytes are read from on demand
pub struct Gguf<S: TensorSource> {
    pub source: S,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    /// file order
    pub tensors: Vec<TensorInfo>,
    by_name: BTreeMap<String, usize>,
    pub alignment: u64,
    pub data_offset: u64,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| anyhow!("GGUF length overflow"))?;
        if end > self.b.len() {
            return Err(NeedMore.into());
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().context("slice length")?)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        let s = self.take(n)?;
        Ok(String::from_utf8(s.to_vec()).context("GGUF string is not UTF-8")?)
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.arr::<1>()?[0]),
            1 => Value::I8(self.arr::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.arr()?)),
            3 => Value::I16(i16::from_le_bytes(self.arr()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.arr()?)),
            6 => Value::F32(f32::from_le_bytes(self.arr()?)),
            7 => {
                let b = self.arr::<1>()?[0];
                if b > 1 {
                    bail!("GGUF bool value {b}");
                }
                Value::Bool(b == 1)
            }
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 0 {
                    bail!("GGUF nested arrays are not part of the format");
                }
                let ety = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(ety, depth + 1)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.arr()?)),
            12 => Value::F64(f64::from_le_bytes(self.arr()?)),
            _ => bail!("GGUF value type {ty} unknown"),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Gguf<FileSource> {
    pub fn open(path: &Path) -> Result<Self> {
        Self::from_source(FileSource::open(path)?).with_context(|| format!("{} is not a readable GGUF", path.display()))
    }
}

struct Header {
    version: u32,
    metadata: BTreeMap<String, Value>,
    tensors: Vec<TensorInfo>,
    alignment: u64,
    data_offset: u64,
}

impl<S: TensorSource> Gguf<S> {
    /// parse the header from a growing prefix of the source; the prefix is dropped afterwards
    pub fn from_source(source: S) -> Result<Self> {
        let file_len = source.len()?;
        let mut want = FIRST_PREFIX as u64;
        let h = loop {
            let n = want.min(file_len) as usize;
            let mut prefix = vec![0u8; n];
            source.read_at(0, &mut prefix)?;
            match parse_header(&prefix) {
                Ok(h) => break h,
                Err(e) if e.is::<NeedMore>() && (n as u64) < file_len => want *= 2,
                Err(e) if e.is::<NeedMore>() => bail!("GGUF truncated: the header runs past the end of the file"),
                Err(e) => return Err(e),
            }
        };
        for t in &h.tensors {
            let end = t.file_offset + t.n_bytes();
            if end > file_len {
                bail!("tensor {} ends at byte {end}, the file has {file_len}", t.name);
            }
        }
        let by_name = h.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { source, version: h.version, metadata: h.metadata, tensors: h.tensors, by_name, alignment: h.alignment, data_offset: h.data_offset })
    }
}

fn parse_header(b: &[u8]) -> Result<Header> {
    let mut c = Cursor { b, pos: 0 };
    if c.take(4)? != MAGIC {
        bail!("no GGUF magic");
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        bail!("GGUF version {version} (2 or 3 expected)");
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let v = c.value(ty, 0).with_context(|| format!("metadata key {key}"))?;
        if metadata.insert(key.clone(), v).is_some() {
            bail!("duplicate metadata key {key}");
        }
    }
    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(Value::U32(a)) => *a as u64,
        Some(v) => bail!("general.alignment has the wrong type: {v:?}"),
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("general.alignment {alignment} is not a power of two");
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    let mut by_name = std::collections::BTreeSet::new();
    for _ in 0..n_tensors as usize {
        let name = c.string()?;
        let n_dims = c.u32()?;
        if n_dims == 0 || n_dims > 4 {
            bail!("tensor {name}: {n_dims} dimensions");
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(c.u64()?);
        }
        let raw_ty = c.u32()?;
        let ty = TensorType::from_ggml(raw_ty).with_context(|| format!("tensor {name}"))?;
        let offset = c.u64()?;
        if offset % alignment != 0 {
            bail!("tensor {name}: offset {offset} not aligned to {alignment}");
        }
        let t = TensorInfo { name: name.clone(), shape, ty, offset, file_offset: 0 };
        if t.shape[0] % t.ty.block().0 != 0 {
            bail!("tensor {name}: row of {} elements is not a whole number of {:?} blocks", t.shape[0], t.ty);
        }
        if !by_name.insert(name.clone()) {
            bail!("duplicate tensor {name}");
        }
        tensors.push(t);
    }
    let data_offset = (c.pos as u64).div_ceil(alignment) * alignment;
    for t in &mut tensors {
        t.file_offset = data_offset + t.offset;
    }
    Ok(Header { version, metadata, tensors, alignment, data_offset })
}

impl<S: TensorSource> Gguf<S> {
    pub fn get(&self, key: &str) -> Result<&Value> {
        self.metadata.get(key).ok_or_else(|| anyhow!("GGUF has no metadata key {key}"))
    }

    pub fn u32(&self, key: &str) -> Result<u32> {
        match self.get(key)? {
            Value::U32(v) => Ok(*v),
            Value::I32(v) if *v >= 0 => Ok(*v as u32),
            v => bail!("{key} is {v:?}, not u32"),
        }
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        match self.get(key)? {
            Value::F32(v) => Ok(*v),
            v => bail!("{key} is {v:?}, not f32"),
        }
    }

    pub fn bool(&self, key: &str) -> Result<bool> {
        match self.get(key)? {
            Value::Bool(v) => Ok(*v),
            v => bail!("{key} is {v:?}, not bool"),
        }
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        match self.get(key)? {
            Value::Str(v) => Ok(v),
            v => bail!("{key} is {v:?}, not a string"),
        }
    }

    pub fn f32_array(&self, key: &str) -> Result<Vec<f32>> {
        match self.get(key)? {
            Value::Array(a) => a
                .iter()
                .map(|v| match v {
                    Value::F32(x) => Ok(*x),
                    v => bail!("{key} holds {v:?}, not f32"),
                })
                .collect(),
            v => bail!("{key} is {v:?}, not an array"),
        }
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        let i = self.by_name.get(name).ok_or_else(|| anyhow!("GGUF has no tensor {name}"))?;
        Ok(&self.tensors[*i])
    }

    /// the type of the weight matrices: the type of the 2-D+ tensors holding the most bytes (norm vectors and small
    /// f32 tensors aside) — what the model "is" (q8_0 text, bf16 vision)
    pub fn matrix_type(&self) -> Result<TensorType> {
        let mut bytes: Vec<(TensorType, u64)> = Vec::new();
        for t in self.tensors.iter().filter(|t| t.shape.len() >= 2) {
            match bytes.iter_mut().find(|(ty, _)| *ty == t.ty) {
                Some((_, b)) => *b += t.n_bytes(),
                None => bytes.push((t.ty, t.n_bytes())),
            }
        }
        bytes.into_iter().max_by_key(|&(_, b)| b).map(|(ty, _)| ty).context("no weight matrices in the GGUF")
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.metadata.contains_key(key)
    }

    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// the bytes of all tensors (what an upload moves; progress reports count against it)
    pub fn data_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.n_bytes()).sum()
    }

    /// a SMALL tensor's raw bytes read into CPU memory (tests, norm vectors) — weights go to the GPU via gpu.rs
    pub fn read_small(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.info(name)?;
        if t.n_bytes() > 1 << 20 {
            bail!("{name}: {} bytes is not small — stream it (Gpu::upload_tensor)", t.n_bytes());
        }
        let mut d = vec![0u8; t.n_bytes() as usize];
        self.source.read_at(t.file_offset, &mut d)?;
        Ok(d)
    }

    /// a small tensor converted to f32 on the CPU
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.info(name)?;
        let d = self.read_small(name)?;
        Ok(match t.ty {
            TensorType::F32 => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            TensorType::F16 => d.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            TensorType::BF16 => d.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            TensorType::Q8_0 => {
                let mut out = Vec::with_capacity(t.n_elements() as usize);
                for blk in d.chunks_exact(34) {
                    let s = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    out.extend(blk[2..].iter().map(|&q| q as i8 as f32 * s));
                }
                out
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }

    /// a tiny GGUF v3: 3 KVs (u32, string array, f32 array) + a bf16 [3,2] tensor and a q8_0 [32] tensor
    fn tiny(alignment: Option<u32>) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(MAGIC);
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend((3 + alignment.is_some() as u64).to_le_bytes());
        s(&mut b, "a.n");
        b.extend(4u32.to_le_bytes());
        b.extend(7u32.to_le_bytes());
        s(&mut b, "a.names");
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        s(&mut b, "x");
        s(&mut b, "yz");
        s(&mut b, "a.mean");
        b.extend(9u32.to_le_bytes());
        b.extend(6u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        for v in [0.5f32, 0.25, -1.0] {
            b.extend(v.to_le_bytes());
        }
        if let Some(a) = alignment {
            s(&mut b, "general.alignment");
            b.extend(4u32.to_le_bytes());
            b.extend(a.to_le_bytes());
        }
        let al = alignment.unwrap_or(32) as u64;
        s(&mut b, "w");
        b.extend(2u32.to_le_bytes());
        b.extend(3u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(30u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        s(&mut b, "q");
        b.extend(1u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(al.to_le_bytes()); // 12 bytes of bf16, next aligned offset
        while b.len() as u64 % al != 0 {
            b.push(0);
        }
        for v in [1.0f32, -2.0, 0.5, 3.0, 0.0, 1.5] {
            b.extend(half::bf16::from_f32(v).to_le_bytes());
        }
        while b.len() as u64 % al != 0 {
            b.push(0);
        }
        b.extend(half::f16::from_f32(0.5).to_le_bytes());
        for i in 0..32i8 {
            b.push((i - 16) as u8);
        }
        b
    }

    #[test]
    fn parses_kvs_tensors_and_data() {
        for al in [None, Some(64)] {
            let g = Gguf::from_source(tiny(al)).unwrap();
            assert_eq!(g.version, 3);
            assert_eq!(g.u32("a.n").unwrap(), 7);
            assert_eq!(g.get("a.names").unwrap(), &Value::Array(vec![Value::Str("x".into()), Value::Str("yz".into())]));
            assert_eq!(g.f32_array("a.mean").unwrap(), vec![0.5, 0.25, -1.0]);
            let w = g.info("w").unwrap();
            assert_eq!((w.shape.clone(), w.ty), (vec![3, 2], TensorType::BF16));
            assert_eq!(g.to_f32("w").unwrap(), vec![1.0, -2.0, 0.5, 3.0, 0.0, 1.5]);
            let q = g.to_f32("q").unwrap();
            assert_eq!(q.len(), 32);
            assert_eq!((q[0], q[31]), (-8.0, 7.5));
            assert_eq!(g.data_offset % al.unwrap_or(32) as u64, 0);
        }
    }

    #[test]
    fn rejects_unknown_type_and_truncation() {
        let mut b = tiny(None);
        // the q tensor's type field: patch Q8_0 (8) → Q4_K (12)
        let pos = b.windows(12).position(|w| w == [1, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]).unwrap() + 12;
        b[pos] = 12;
        assert!(format!("{:#}", Gguf::from_source(b).err().unwrap()).contains("not supported"));
        let mut t = tiny(None);
        t.truncate(t.len() - 1);
        assert!(Gguf::from_source(t).is_err());
        assert!(Gguf::from_source(b"GGUX".to_vec()).is_err());
    }
}
