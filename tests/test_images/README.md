# Test Images

This directory should contain sample images for testing the style transfer API.

## Required Files

- `subject.jpg` - A subject image (e.g., portrait, landscape, object)
- `style.jpg` - A style reference image (e.g., artwork, painting)

## Image Guidelines

- **Format**: JPEG or PNG
- **Size**: Recommended 512x512 to 1024x1024 pixels
- **File Size**: Under 10MB each
- **Content**: Use clear, high-quality images for best results

## Example Sources

- **Subject Images**: Personal photos, stock photography, portraits
- **Style Images**: Famous paintings, artistic styles, textures

## Usage

The test scripts (`test_api.py` and `load_test.py`) will automatically use these images when running tests.

```bash
# Add your test images
cp your_subject_image.jpg test_images/subject.jpg
cp your_style_image.jpg test_images/style.jpg

# Run tests
python test_api.py
python load_test.py
```