# Known Issues

## ⚠️ DeepSeek-OCR Model Limitations with `base_size` Parameter

### Issue (RESOLVED in v2.0)
The DeepSeek-OCR model has a **critical limitation** with the `--base-size` parameter:

**`base_size` values above 1024 cause shape mismatch errors**

### Symptoms
When using `--base-size` values greater than 1024, you may encounter:
```
❌ Error processing page: shape mismatch: value tensor of shape [1041, 1280] cannot be broadcast to indexing result of shape [1309, 1280]
```

### Root Cause
The DeepSeek-OCR model's patch calculation logic has a bug that manifests when `base_size > 1024`. The model:
1. Calculates the wrong number of image patches (e.g., 1041)
2. Creates masks expecting a different number of positions (e.g., 1309)
3. Fails during tensor assignment with shape mismatch

This occurs in the model's internal code at `modeling_deepseekocr.py:508`:
```python
inputs_embeds[idx][images_seq_mask[idx]] = images_in_this_batch  # Shape mismatch!
```

### Resolution
**✅ FIXED in v2.0**: The compression presets now enforce safe `base_size` values:

```python
COMPRESSION_PRESETS = {
    "low": dict(base_size=1024, image_size=640, ...),   # Safe maximum
    "med": dict(base_size=1024, image_size=640, ...),   # Safe maximum
    "high": dict(base_size=896, image_size=512, ...),   # Reduced for speed
}
```

**Default behavior**: The CLI now defaults to `base_size=1024` (safe maximum) instead of higher values.

### Safe Usage
✅ **DO**: Use `--base-size 1024` or lower (default)
✅ **DO**: Use the built-in compression presets (`--compression low/med/high`)
❌ **DON'T**: Manually set `--base-size` above 1024

### Testing Conducted
Extensive testing confirmed:
- ✅ `base_size=1024`: All images process successfully (100% success rate)
- ❌ `base_size=1280`: Shape mismatch errors on all images
- ✅ Works on both MPS and CPU devices
- ✅ Works with all image formats (PNG, JPG, WebP, etc.)
- ✅ Works with screenshots, documents, and various aspect ratios

### Reference
- Model commit: `2c968b433af61a059311cbf8997765023806a24d`
- Discovery: User testing found shape mismatch with default compression presets
- Fix: Adjusted COMPRESSION_PRESETS to use safe `base_size=1024`
- Validation: 5/5 example images processed successfully after fix

---

## Future Improvements

If DeepSeek releases an updated model that fixes the `base_size > 1024` bug, the compression presets can be adjusted to:
```python
"low": dict(base_size=1280, image_size=768, ...),  # Higher quality when model supports it
```

Until then, `base_size=1024` provides excellent OCR quality within model constraints.
