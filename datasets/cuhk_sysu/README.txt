## Cite

@article{xiao2016end
  title={End-to-End Deep Learning for Person Search},
  author={Xiao, Tong and Li, Shuang and Wang, Bochao and Lin, Liang and Wang, Xiaogang},
  journal={arXiv:1604.01850},
  year={2016}
}


## Terms of Use

By downloading the dataset, you agree to the following terms:

1.  You will use the data only for non-commercial research and educational purposes.
2.  You will **NOT** distribute the dataset.
3.  The Chinese University of Hong Kong makes no representations or warranties regarding the data. All rights of the images reserved by the original owners.
4.  You accept full responsibility for your use of the data and shall defend and indemnify The Chinese University of Hong Kong, including their employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.


## Overall Information

The dataset is collected from two sources: street snap and movie.
In street snap, 12,490 images and 6,057 query persons were collected
with movable cameras across hundreds of scenes while 5,694 images and
2,375 query persons were selected from movies and TV dramas.

We provide notations for both person re-identification and pedestrian
detection. The data is partitioned into a training set and a test set.
The training set contains 11,206 images and 5,532 query persons.
The test set contains 6,978 images and 2,900 query persons. The
training and test sets have no overlap on images and query persons.
We also construct several subsets for evaluating the influence of various
factors on person search.


## Introduction

'./Image/SSM':              Images collected from street snap and movies.

'./annotation/Images.mat':  1*18184 struct (18184 images)
                 Each line describes the pedestrian information of an image,
                 including the image name (imname), the number and locations
                 of pedestrians appearing (nAppear and box) in this image.
'./annotation/Person.mat':  1*11934 struct (11934 persons with each person shows up in at least two images)
                 The person information includs the person id (idname), appearing time (nAppear),
                 and location in each scene (Person(i).scene.idlocate).
'./annotation/pool.mat':    6978 test images.

'./annotation/test/train_test/Train.mat':             5532 query persons for training.
'./annotation/test/train_test/TestG50-TestG4000.mat': 2900 query persons with gallery size varies from 50 to 4000 for testing.
'./annotation/test/subset/Occlusion.mat':             187 query persons with occlusion.
'./annotation/test/subset/Resolution.mat':            290 query persons with low resolution.

*Note: The location of each person is stored as (xmin, ymin, width, height),
i.e. crop_im = I ( idlocate(2):idlocate(2)+idlocate(4), idlocate(1):idlocate(1)+idlocate(3) );


