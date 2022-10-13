# Captcha recognition
 _____________________________

Last update 13.10.22
______________________________

## Data 
### From open sources
- [source2](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)
  - The images are 5 letter words that can contain numbers. 
    The images have had noise applied to them (blur and a line). They are 200 x 50 PNGs. \
    They are in Greyscale but be aware that they still have 3Dimensions
- [source3](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images)
  - Captcha Images with different styles, fonts, colors. \
    These images consist of 10-character strings with randomized colors and alignment. \
    The filenames are the text present in the captcha images. Currently, there are 10000 images.
- [source4](https://www.kaggle.com/datasets/codingnirvana/captcha-images)
  - 6 letters (only letters)
- [source6](https://www.kaggle.com/datasets/akashguna/large-captcha-dataset)
  - Large Scale Captcha Dataset containing 82.3 K images of different 5 characted captchas.
- [source7](https://www.kaggle.com/datasets/kiranbudati/1-million-captcha-images)
  - 1 Million Synthetic Captcha Images

### To generate your own synthetic dataset

Run 

`python captcha/data_utils/generate_synthetic_data.py data/raw/synthetic_test 10`

---------------------------
## Model weights

You can download last release using [this](https://drive.google.com/drive/folders/1WP0bo3NF172F908lmOYTjIz7mxRTIGuw?usp=sharing) Google Drive link
___________________________

## API
Run uvicorn from captcha/api directory

`uvicorn main:app --reload`

### Request

1. go to `http://127.0.0.1:8000/docs`

upload image and get the response: 
```
{
  "endpoint_name": "predict",
  "predictions": [
    [4, 7, 7, 9, 5, 8],
  ],
  "labels": [
    [4, 2, 7, 9, 6, 8],
  ],
  "decode_prediction": [
    ['4', '7', '7', '9', '5', '8'],
  ],
  "decode_labels": [
    ['4', '2', '7', '9', '6', '8'],
  ],
  "other": null,
  "message": "Successful"
}
```

or

2. Send request using python

```
response = requests.post(
        url="http://127.0.0.1:8000/predict",
        files=[("data", open(file_path, "rb"))],
    )
```

or

3. Use curl

`curl -X POST "http://127.0.0.1:8000/predict"  -F "data=@tests/test_data/000037.png"`

----------------
# Docker API
Run

`docker build -t captcha_breaker .`

`docker run -d --name vsem_pizda -p 80:80 captcha_breaker`


## TODO

- scheduler Cosine Annealing
- Resnet50 (Resnet-D, Resnest, Resnext)
- Adam + warmup
- Mixed Precision
- dvc for splits
- profile
- Perform augs on the GPU (Kornia, DALI)
- hyperparameter tuning
- Rabbit