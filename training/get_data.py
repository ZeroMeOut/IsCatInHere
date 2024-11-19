import requests
image_url = 'https://cataas.com/cat'

for i in range(1000):
    img_data = requests.get(image_url).content
    with open(f'dataset/cats/cat{i}.jpg', 'wb') as handler:
        handler.write(img_data)