import os
import requests

# Create directories if they don't exist
os.makedirs('Mydataset/train/cats', exist_ok=True)
os.makedirs('Mydataset/val/cats', exist_ok=True)

cat_url = 'https://cataas.com/cat'
datacount = 1000
traincount = int(datacount * 0.8)

for i in range(datacount):
    try:
        # Download cat image
        cat_data = requests.get(cat_url).content
        if i < traincount:
            with open(f'Mydataset/train/cats/cat{i}.jpg', 'wb') as handler:
                handler.write(cat_data)
        else:
            with open(f'Mydataset/val/cats/cat{i}.jpg', 'wb') as handler:
                handler.write(cat_data)
            
        # Print progress every 50 images
        if (i + 1) % 50 == 0:
            print(f'Downloaded {i + 1} images')
            
    except Exception as e:
        print(f'Error downloading image {i}: {str(e)}')
        continue

print('Download complete!')