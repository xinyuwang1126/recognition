import os
import numpy as np
import h5py
import json
import torch
from imageio import imread
from PIL import Image
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(train_json,  dev_json, test_json, image_folder, captions_per_image,  output_folder, max_len
                       ):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """


    # Read JSON
    with open(train_json, 'r') as j:
        train_data = json.load(j)

    with open(dev_json, 'r') as j:
        dev_data = json.load(j)

    with open(test_json, 'r') as j:
        test_data = json.load(j)


    # Read image paths and captions for each image
    tokens = Counter()
    role2id = Counter()
    train_image_paths = []
    train_image_frames = []
    train_image_roles = []
    dev_image_paths = []
    dev_image_frames = []
    dev_image_roles = []
    test_image_paths = []
    test_image_frames = []
    test_image_roles = []

    for key in train_data:
        frames = train_data[key]['frames']
        for frame in frames:
            values = list(frame.values())
            tokens.update(values)
        roles = list(frames[0].keys())
        role2id.update(roles)
        verb = train_data[key]['verb']
        path = os.path.join(image_folder,'train', verb, key)
        train_image_paths.append(path)
        train_image_frames.append(frames)
        train_image_roles.append(roles)

    assert len(train_image_paths) == len(train_image_frames)
    assert len(train_image_paths) == len(train_image_roles)

    for key in dev_data:
        frames = dev_data[key]['frames']
        for frame in frames:
            values = list(frame.values())
            tokens.update(values)
        roles = list(frames[0].keys())
        role2id.update(roles)
        verb = dev_data[key]['verb']
        path = os.path.join(image_folder,'dev', verb, key)
        dev_image_paths.append(path)
        dev_image_frames.append(frames)
        dev_image_roles.append(roles)

    assert len(dev_image_paths) == len(dev_image_frames)
    assert len(dev_image_paths) == len(dev_image_roles)

    for key in test_data:
        frames = test_data[key]['frames']
        for frame in frames:
            values = list(frame.values())
            tokens.update(values)
        roles = list(frames[0].keys())
        role2id.update(roles)
        verb = test_data[key]['verb']
        path = os.path.join(image_folder,'test', verb, key)
        test_image_paths.append(path)
        test_image_frames.append(frames)
        test_image_roles.append(roles)

    assert len(test_image_paths) == len(test_image_frames)
    assert len(test_image_paths) == len(test_image_roles)



    # Create word map
    values = [w for w in tokens.keys()]
    token2id = {k: v + 1 for v, k in enumerate(values)}
    token2id['<unk>'] = len(token2id) + 1
    token2id['<pad>'] = 0
    role_values = [w for w in role2id.keys()]
    roles2id = {k: v +1  for v, k in enumerate(role_values)}
    roles2id['<pad>'] = 0

    # Create a base/root name for all output files
    #base_filename = ''

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'token2id'  + '.json'), 'w') as j:
        json.dump(token2id, j, indent = 4)

    with open(os.path.join(output_folder, 'roles2id'  + '.json'), 'w') as j:
        json.dump(roles2id, j, indent = 4)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    print('start iteration')
    for impaths, imcaps, roles, split in [(train_image_paths, train_image_frames, train_image_roles, 'TRAIN'),
                                   (dev_image_paths, dev_image_frames, dev_image_roles, 'VAL'),
                                   (test_image_paths, test_image_frames, test_image_roles, 'TEST')]:

        #with h5py.File(os.path.join(output_folder, split + '_IMAGES.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
           # h.attrs['frames_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            #images = h.create_dataset('images', (len(impaths)-2, 3, 256, 256), dtype='uint8')
            #images = h.create_dataset('images', (10, 3, 256, 256), maxshape=(len(impaths),3,256,256),dtype='uint8')
        print("\nReading %s images and frames, storing to file...\n" % split)

        enc_captions = []
        caplens = []
        all_enc_roles = []
        count = 0
        for i, path in enumerate(impaths):
            try:
                img = imread(impaths[i])
                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                    #img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                # img = imresize(img, (256, 256))
                img = np.array(Image.fromarray(img).resize((256,256)))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                #images[count] = img
                name = path.split('/')[-1]
                name = name[:-4]
                #print(name)
                filepath=os.path.join('/tmp/xinyuw3/input_file',split,name)
                #print(path)
                np.save(filepath,img)
               
                #print(count)
                count+=1

                semantic_roles = roles[i]
                enc_roles = [roles2id[k] for k in semantic_roles]
                assert len(semantic_roles) == len(enc_roles)

                if len(enc_roles) < max_len:
                    enc_roles = enc_roles + [roles2id['<pad>']] * (max_len - len(enc_roles))
                all_enc_roles.append(enc_roles)
                assert len(enc_roles) == max_len                   


                for j, c in enumerate(captions):
                    # Encode captions
                        #print(c)
                    enc_c = []
                    for k in semantic_roles:
                        enc_c.append(token2id[c[k]])

                    if len(enc_c) != len(semantic_roles):
                        print(path)                       
                    assert len(enc_c) == len(semantic_roles)
                    c_len = len(enc_c)
                        #print(enc_c)
                    if len(enc_c) < max_len:
                        enc_c = enc_c + [token2id['<pad>']] * (max_len - len(enc_c))
                    assert len(enc_c) == max_len
                    # Find frame lengths
                        #c_len = len(c)

                    enc_captions.append(enc_c)
                    caplens.append(c_len)
            except:
                #break
                #pass
                print(path)
                pass
                    #break
        print(count, len(enc_captions), len(caplens), len(all_enc_roles))
            # Sanity check
        assert count * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_FRAMES'  + '.json'), 'w') as j:
            json.dump(enc_captions, j, indent = 4)

        with open(os.path.join(output_folder, split + '_FRAMELENS' + '.json'), 'w') as j:
            json.dump(caplens, j, indent = 4)

        with open(os.path.join(output_folder, split + '_ROLES' + '.json'), 'w') as j:
            json.dump(all_enc_roles, j, indent = 4)

train_json='/tmp/xinyuw3/imsitu/train.json'
dev_json='/tmp/xinyuw3/imsitu/dev.json'
test_json='/tmp/xinyuw3/imsitu/test.json'
image_path='/tmp/xinyuw3/imsitu/'
#train_json='/data/MM21/data/imsitu/train.json'
#dev_json='/data/MM21/data/imsitu/dev.json'
#test_json='/data/MM21/data/imsitu/test.json'
#image_path='/data/MM21/data/imsitu/'

create_input_files(train_json,  dev_json, test_json, image_path, 3,  '/tmp/xinyuw3/input_file/', 6
                       )
