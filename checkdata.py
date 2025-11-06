import os

root = r'D:\DACN\dataset\Flickr8k'
token = os.path.join(root, 'Flickr8k_text', 'Flickr8k.token.txt')
print('token exists:', os.path.exists(token))
img_dirs = ['images']
for d in img_dirs:
    p = os.path.join(root, d)
    print(p, 'exists:', os.path.exists(p), 'num files:', len([f for f in os.listdir(p) if f.lower().endswith('.jpg')]) if os.path.exists(p) else 0)

# check one example reported missing
example = os.path.join(root, 'Flickr_8k_Dataset', '1000268201_693b08cb0e.jpg')
print('example file exists:', os.path.exists(example))

# count how many caption images are missing
missing = []
if os.path.exists(token):
    with open(token, 'r', encoding='utf-8') as f:
        for line in f:
            img = line.split('\t')[0].split('#')[0]
            img_path = os.path.join(root, 'Flickr_8k_Dataset', img)
            if not os.path.exists(img_path):
                missing.append(img)
    print('missing images referenced in token (sample 20):', missing[:20])
    print('total referenced images:', len(set([line.split("\t")[0].split("#")[0] for line in open(token, 'r', encoding="utf-8")])))
    print('total missing referenced images:', len(missing))
else:
    print('token file not found; cannot check referenced images.')