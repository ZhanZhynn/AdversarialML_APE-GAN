# Google Colab Link
https://colab.research.google.com/drive/1F_drQCAzPNj49HfQAZfNCo4F07DrMmdE?usp=sharing

# APE-GAN
Adversarial Perturbation Elimination with GAN
My Implementation of APE-GAN with reference to the research paper (https://arxiv.org/pdf/1707.05474.pdf)

## MNIST

![MNIST](https://github.com/owruby/APE-GAN/blob/master/MNIST.png)

### 1. Train CNN and Generate Adversarial Examples(FGSM)
```
python generate.py --eps 0.15 ./checkpoint/mnist
```

### 2. Train APE-GAN
```
python train.py --checkpoint ./checkpoint/mnist
```

### 3. Test
```
python test_model.py --eps 0.15 --gan_path ./checkpoint/mnist/3.tar
```

## CIFAR-10

![CIFAR10](https://github.com/owruby/APE-GAN/blob/master/CIFAR10.png)

### 1. Train CNN and Generate Adversarial Examples(FGSM)
```
python generate.py --data cifar --eps 0.01
```

### 2. Train APE-GAN
```
python train.py --data cifar --epochs 30 --checkpoint ./checkpoint/cifar
```

### 3. Test
```
python test_model.py --data cifar --eps 0.01 --gan_path ./checkpoint/cifar/10.tar
```
