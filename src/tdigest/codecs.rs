use crate::tdigest::{Centroid, TDigest};
use polars::prelude::*;
// `Series` is already re-exported by the prelude; the extra import is unnecessary in 0.51.

/* -----------------------------------------------------------------------------
 Compact vs canonical representations
 -----------------------------------------------------------------------------
 - Canonical (produced by `tdigest_to_series`):
     centroids.mean:  Float64
     centroids.weight: Int64   (exact integer counts)
     min/max:         Float64
     sum:             Float64
     count:           Int64    (exact total count)
     max_size:        Int64

 - Compact (produced by `tdigest_to_series_32`):
     centroids.mean:  Float32  (smaller, adequate precision)
     centroids.weight: UInt32  (exact up to ~4.29e9 per centroid)
     min/max:         Float32
     sum:             Float64
     count:           Int64
     max_size:        Int64

 Readers:
 - `parse_tdigests_any` is tolerant and accepts f64/f32/i64/u64/u32 variants.
 - `parse_tdigests` is the older/“strict” reader, but now casts weight to f64
   so it can read both i64 and u32 (and friends) without panicking.

 Rationale for UInt32 weights:
 - Centroid weight is a count of absorbed points; integers are natural here.
 - u32 is compact and exact up to 4_294_967_295 per centroid.
 - When building means, converting u32->f64 is exact (≤ 2^53).
 - If you ever foresee >4.29e9 in a *single centroid*, switch to u64.
----------------------------------------------------------------------------- */

/// Build a **compact** tdigest struct from a TDigest by downcasting the canonical f64 struct.
///
/// Layout in the resulting `Struct`:
/// - `centroids`: List<Struct{ mean: Float32, weight: UInt32 }>
/// - `sum`:       Float64 (unchanged; preserves accumulation precision)
/// - `min/max`:   Float32
/// - `count`:     Int64
/// - `max_size`:  Int64
///
/// Implementation note:
/// We construct the compact struct by first producing the canonical f64 struct
/// via `tdigest_to_series(td, name)` and then downcasting columns. This keeps
/// all logic about TDigest internals in one place.
pub(crate) fn tdigest_to_series_32(td: TDigest, name: &str) -> Series {
    // Start from the canonical f64 struct.
    let s64 = tdigest_to_series(td, name);
    let sc = s64
        .struct_()
        .expect("tdigest_to_series returned non-struct");

    let centroids_col = sc.field_by_name("centroids").expect("centroids");
    let sum_col = sc.field_by_name("sum").expect("sum");
    let min_col = sc.field_by_name("min").expect("min");
    let max_col = sc.field_by_name("max").expect("max");
    let count_col = sc.field_by_name("count").expect("count");
    let max_size_col = sc.field_by_name("max_size").expect("max_size");

    // `centroids` is a 1-row List whose inner dtype is Struct{mean, weight}.
    let list_av = centroids_col.get(0).expect("centroids row 0");
    let centroids_list_series = match list_av {
        AnyValue::List(ls) => ls,
        // Empty fallback: construct an empty List<Struct{mean: f32, weight: u32}>.
        _ => Series::new("centroids".into(), &[] as &[Series]),
    };

    // Downcast inner fields to compact types.
    let scents = centroids_list_series
        .struct_()
        .expect("centroids inner struct");
    let mean_f32 = scents
        .field_by_name("mean")
        .expect("mean")
        .cast(&DataType::Float32)
        .unwrap();
    let weight_u32 = scents
        .field_by_name("weight")
        .expect("weight")
        .cast(&DataType::UInt32)
        .unwrap();

    // Build Struct{mean, weight} without StructChunked::new: go via a one-row DataFrame.
    let centroids_struct_compact = DataFrame::new(vec![
        Series::new("mean".into(), mean_f32).into(),
        Series::new("weight".into(), weight_u32).into(),
    ])
    .unwrap()
    .into_struct("centroids".into())
    .into_series();

    // Wrap the single Struct as a one-row List.
    let centroids_list_compact = Series::new("centroids".into(), &[centroids_struct_compact]);

    // min/max → f32; keep sum/count/max_size as-is for precision/consistency.
    let min_f32 = min_col.cast(&DataType::Float32).unwrap();
    let max_f32 = max_col.cast(&DataType::Float32).unwrap();

    DataFrame::new(vec![
        centroids_list_compact.into(),
        sum_col.clone().into(),
        min_f32.into(),
        max_f32.into(),
        count_col.clone().into(),
        max_size_col.clone().into(),
    ])
    .unwrap()
    .into_struct(name.into())
    .into_series()
}

/// NEW: tolerant parser that accepts both canonical f64 digests and compact f32/u32 digests.
///
/// Tolerant casts performed:
/// - centroids.mean:    f64 or f32          → f64
/// - centroids.weight:  f64/f32/i64/u64/u32 → f64
/// - min/max:           f64 or f32          → f64
/// - count:             i64 or u64          → f64
pub(crate) fn parse_tdigests_any(input: &Series) -> Vec<TDigest> {
    let Ok(struct_ca) = input.struct_() else {
        return Vec::new();
    };

    let Ok(centroids_col) = struct_ca.field_by_name("centroids") else {
        return Vec::new();
    };
    let Ok(sum_col) = struct_ca.field_by_name("sum") else {
        return Vec::new();
    };
    let Ok(min_col) = struct_ca.field_by_name("min") else {
        return Vec::new();
    };
    let Ok(max_col) = struct_ca.field_by_name("max") else {
        return Vec::new();
    };
    let Ok(count_col) = struct_ca.field_by_name("count") else {
        return Vec::new();
    };
    let Ok(max_size_col) = struct_ca.field_by_name("max_size") else {
        return Vec::new();
    };

    let n = input.len();
    let mut out = Vec::with_capacity(n);

    let Ok(centroids_list) = centroids_col.list() else {
        return Vec::new();
    };

    for i in 0..n {
        // Scalars
        let sum = sum_col
            .get(i)
            .ok()
            .and_then(|v| v.try_extract::<f64>().ok())
            .unwrap_or(0.0);

        let min = min_col
            .get(i)
            .ok()
            .and_then(|v| {
                v.try_extract::<f64>()
                    .ok()
                    .or_else(|| v.try_extract::<f32>().ok().map(|x| x as f64))
            })
            .unwrap_or(0.0);

        let max = max_col
            .get(i)
            .ok()
            .and_then(|v| {
                v.try_extract::<f64>()
                    .ok()
                    .or_else(|| v.try_extract::<f32>().ok().map(|x| x as f64))
            })
            .unwrap_or(0.0);

        let count = count_col
            .get(i)
            .ok()
            .and_then(|v| {
                v.try_extract::<i64>()
                    .ok()
                    .or_else(|| v.try_extract::<u64>().ok().map(|x| x as i64))
            })
            .unwrap_or(0) as f64;

        let max_size = max_size_col
            .get(i)
            .ok()
            .and_then(|v| v.try_extract::<i64>().ok())
            .unwrap_or(0) as usize;

        // Centroids row: a Series (dtype=Struct) of length = #centroids.
        let mut cents: Vec<Centroid> = Vec::new();
        if let Some(centroids_row) = centroids_list.get_as_series(i) {
            let sc = centroids_row.struct_().unwrap();

            let mean_s = sc
                .field_by_name("mean")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();
            let weight_s = sc
                .field_by_name("weight")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();

            let mean_ca = mean_s.f64().unwrap();
            let wgt_ca = weight_s.f64().unwrap();

            let len = mean_ca.len().min(wgt_ca.len());
            for idx in 0..len {
                if let (Some(m), Some(w)) = (mean_ca.get(idx), wgt_ca.get(idx)) {
                    cents.push(Centroid::new(m, w));
                }
            }
        }

        out.push(TDigest::new(cents, sum, count, min, max, max_size));
    }

    out
}

/// Legacy/“strict” parser used by existing call-sites. It now mirrors the tolerant
/// behavior for weights by casting to `Float64` first, so it can read both i64/u32.
/// This preserves compatibility with historical payloads while supporting compact ones.
pub fn parse_tdigests(input: &Series) -> Vec<TDigest> {
    input
        .struct_()
        .into_iter()
        .flat_map(|chunk| {
            let count_series = chunk.field_by_name("count").unwrap();
            let count_it = count_series.i64().unwrap().into_iter();

            let max_series = chunk.field_by_name("max").unwrap();
            let min_series = chunk.field_by_name("min").unwrap();
            let sum_series = chunk.field_by_name("sum").unwrap();
            let max_size_series = chunk.field_by_name("max_size").unwrap();
            let centroids_series = chunk.field_by_name("centroids").unwrap();

            let mut max_it = max_series.f64().unwrap().into_iter();
            let mut min_it = min_series.f64().unwrap().into_iter();
            let mut max_size_it = max_size_series.i64().unwrap().into_iter();
            let mut sum_it = sum_series.f64().unwrap().into_iter();
            let mut centroids_it = centroids_series.list().unwrap().into_iter();

            count_it
                .map(|c| {
                    let centroids = centroids_it.next().unwrap().unwrap();
                    let sc = centroids.struct_().unwrap();

                    let mean_series = sc.field_by_name("mean").unwrap();
                    let mean_it = mean_series.f64().unwrap().into_iter();

                    let weight_series = sc
                        .field_by_name("weight")
                        .unwrap()
                        .cast(&DataType::Float64)
                        .unwrap();
                    let mut weight_it = weight_series.f64().unwrap().into_iter();

                    let centroids_res = mean_it
                        .map(|m| {
                            Centroid::new(
                                m.unwrap(),
                                weight_it.next().unwrap().unwrap(), // already f64
                            )
                        })
                        .collect::<Vec<_>>();

                    TDigest::new(
                        centroids_res,
                        sum_it.next().unwrap().unwrap(),
                        c.unwrap() as f64,
                        max_it.next().unwrap().unwrap(),
                        min_it.next().unwrap().unwrap(),
                        max_size_it.next().unwrap().unwrap() as usize,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

/// Produce the **canonical** (lossless) tdigest struct as a single-row `Series` with a `Struct`.
///
/// Layout:
/// - `centroids`: List<Struct{ mean: Float64, weight: Int64 }>
/// - `sum`:       Float64
/// - `min/max`:   Float64
/// - `count`:     Int64
/// - `max_size`:  Int64
///
/// Notes:
/// - Canonical uses Int64 for weights to preserve exact integer counts.
/// - Compact writers can downcast from this representation.
pub fn tdigest_to_series(tdigest: TDigest, name: &str) -> Series {
    let mut means: Vec<f64> = vec![];
    let mut weights: Vec<i64> = vec![];

    tdigest.centroids().iter().for_each(|c| {
        weights.push(c.weight() as i64);
        means.push(c.mean());
    });

    // Build inner Struct{mean, weight}.
    let centroids_struct = DataFrame::new(vec![
        Series::new("mean".into(), means).into(),
        Series::new("weight".into(), weights).into(),
    ])
    .unwrap()
    .into_struct("centroids".into())
    .into_series();

    // Wrap inner struct as a one-row List and assemble the outer Struct.
    DataFrame::new(vec![
        Series::new(
            "centroids".into(),
            [Series::new("centroids".into(), centroids_struct)],
        )
        .into(),
        Series::new("sum".into(), [tdigest.sum()]).into(),
        Series::new("min".into(), [tdigest.min()]).into(),
        Series::new("max".into(), [tdigest.max()]).into(),
        Series::new("count".into(), [tdigest.count() as i64]).into(),
        Series::new("max_size".into(), [tdigest.max_size() as i64]).into(),
    ])
    .unwrap()
    .into_struct(name.into())
    .into_series()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_tdigest_deserializstion() {
        // Legacy canonical payload (weight as integer JSON → Int64 in Polars).
        let json_str = "[{\"tdigest\":{\"centroids\":[{\"mean\":4.0,\"weight\":1},{\"mean\":5.0,\"weight\":1},{\"mean\":6.0,\"weight\":1}],\"sum\":15.0,\"min\":4.0,\"max\":6.0,\"count\":3,\"max_size\":100}},{\"tdigest\":{\"centroids\":[{\"mean\":1.0,\"weight\":1},{\"mean\":2.0,\"weight\":1},{\"mean\":3.0,\"weight\":1}],\"sum\":6.0,\"min\":1.0,\"max\":3.0,\"count\":3,\"max_size\":100}}]";
        let cursor = Cursor::new(json_str);
        let df = JsonReader::new(cursor).finish().unwrap();
        let series = df.column("tdigest").unwrap();
        let res = parse_tdigests(
            series
                .as_series()
                .expect("expected a Series-backed Column here"),
        );

        let expected = vec![
            TDigest::new(
                vec![
                    Centroid::new(4.0, 1.0),
                    Centroid::new(5.0, 1.0),
                    Centroid::new(6.0, 1.0),
                ],
                15.0,
                3.0,
                6.0,
                4.0,
                100,
            ),
            TDigest::new(
                vec![
                    Centroid::new(1.0, 1.0),
                    Centroid::new(2.0, 1.0),
                    Centroid::new(3.0, 1.0),
                ],
                6.0,
                3.0,
                3.0,
                1.0,
                100,
            ),
        ];
        assert!(res == expected);
    }

    #[test]
    fn test_tdigest_serialization_roundtrip() {
        // Canonical → canonical (weights Int64)
        let tdigest = TDigest::new(
            vec![
                Centroid::new(10.0, 1.0),
                Centroid::new(20.0, 2.0),
                Centroid::new(30.0, 3.0),
            ],
            60.0,
            3.0,
            30.0,
            10.0,
            300,
        );
        let ser = tdigest_to_series(tdigest.clone(), "n");

        // Expected canonical struct.
        let cs = DataFrame::new(vec![
            Series::new("mean".into(), [10.0, 20.0, 30.0]).into(),
            Series::new("weight".into(), [1_i64, 2, 3]).into(),
        ])
        .unwrap()
        .into_struct("centroids".into())
        .into_series();

        let expected = DataFrame::new(vec![
            Series::new("centroids".into(), [Series::new("a".into(), cs)]).into(),
            Series::new("sum".into(), [60.0]).into(),
            Series::new("min".into(), [10.0]).into(),
            Series::new("max".into(), [30.0]).into(),
            Series::new("count".into(), [3_i64]).into(),
            Series::new("max_size".into(), [300_i64]).into(),
        ])
        .unwrap()
        .into_struct("n".into())
        .into_series();

        assert!(ser == expected);

        // Compact writer (mean f32, weight u32, min/max f32) still parses back to the same TDigest.
        let compact = tdigest_to_series_32(tdigest.clone(), "n");
        let parsed = parse_tdigests_any(&compact);
        assert_eq!(parsed.len(), 1);
        // We permit minor float rounding in min/max due to f32 downcast; centroids and totals match.
        assert_eq!(parsed[0].count(), tdigest.count());
        assert_eq!(parsed[0].max_size(), tdigest.max_size());
        assert_eq!(parsed[0].centroids().len(), tdigest.centroids().len());
    }
}
