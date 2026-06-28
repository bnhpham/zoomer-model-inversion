# Zoomer

An executable is given as part of an authentication system and contains a model that is pretrained on ImageNet and fine-tuned on 18 authorized samples. The objective of this challenge is to recover these 18 training samples, which is known as a model inversion attack.

## Reverse-Engineering the Executable

To gain more knowledge about the target model, we used Ghidra, a reverse-engineering tool that converts compiled binaries back into human-readable code. The decompiled `build_and_execute` function of the given authentication system shows that it uses a ResNet-18-like model as a binary classifier. Moreover, we found that the creator of the file had defined a section called `.weights`, which holds all the model parameters. Using Ghidra's memory map, we discovered that the section begins at file offset `0x13A00` and has a total size of `0x2AAC800`. The structure of this section is documented in the table `weight_index` found in the decompiled `get_weight` function. This table consists of `0x66` = 102 rows, where each contains the name of the DNN component to which the parameters belong, the file offset and the size of the component. Furthermore, each table entry has a size of 8 bytes, whereas the weights themselves are stored as 4-byte floating-points. To reconstruct the target model, we first convert the virtual address of the `weight_index` table into a file offset. We then iterate through the table, extract the corresponding parameters from the ".weights" section using the offsets and lengths and load them into PyTorch's ResNet-18 implementation. Using the obtained model, we can now perform a white-box model inversion attack.

## DeepInversion

### Concept

To recover the training data, we implement an approach named "DeepInversion" proposed by [Yin et al.](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html), which exploits the statistics stored in the BatchNorm (BN) layers of the target model. A BN layer normalizes the output (also called activation) of the previous layer to have zero mean and unit variance per channel, making training faster and stable. They track the mean and variance across all training batches and continuously update them. After training, these statistics are frozen and used at inference time to normalize new inputs. Although intended solely for normalization, they also reveal how the training data looks like statistically at every BN layer. DeepInversion exploits this observation by generating a batch of images that matches the training data at each layer in terms of mean and variance.

The method optimizes pixel values of random noise images $\hat{x}$ for a specific target class by [gradient descent (line 103-200)](src/deep_inversion.py). In each optimization step, we apply [random jitter and horizontal flipping (line 145)]() as augmentation techniques to encourage generating solid objects. We then let the target model predict the synthesized images and compute a loss specified as follows:

$$L_{total} = L_{class} + \alpha_{BN} L_{BN} + \text{Image prior}$$

$L_{class}$ is the classification loss implemented as the cross-entropy. $L_{BN}$, on the other hand, describes the sum of [distances between current activations and saved statistics across all BN layers (line 75-99)](src/deep_inversion.py) scaled by $\alpha_{BN}$. The [image prior (line 165)](src/deep_inversion.py) is a collection of regularization terms to make the images smooth rather than noisy. It is defined as the following: 

$$\text{Image prior} = \alpha_{TV2} L_{TV2} + \alpha_{TV1} L_{TV1} + \alpha_{l_2} ||\hat{x}||_2$$

The total variation (TV) loss measures how much adjacent pixels differ from each other. If adjacent pixels are very different, the image looks noisy. Therefore, minimizing the TV loss w.r.t. the $l_2$ and $l_1$ distance encourages smoothness. $||x||_2$ describes the $l_2$ norm of the generated image. $\alpha_{TV2}$, $\alpha_{TV1}$ and $\alpha_{l_2}$ are corresponding scaling factors.

### Adaptive DeepInversion

To increase the diversity of the synthesized images, [Yin et al.](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html) propose an extension of DeepInversion called "Adaptive DeepInversion". Alongside optimization, a student model is trained on the images in parallel such that it memorizes all images that have been generated so far. Furthermore, a [competition loss (line 50-69)](src/deep_inversion.py) is added that penalizes the images if the student mimics the teacher (i.e. the target model) well. More specifically, it maximizes the Jensen-Shannon (JS) divergence between the outputs of the student and teacher, which encourages more diverse images that the student hasn't learned yet and thus results in a broader coverage of the training set. The new loss function is specified as the following:

$$L_{total} = L_{class} + \alpha_{BN} L_{BN} + \text{Image prior} + \alpha_{c} L_{compete}$$

$L_{compete}$ is defined as $1 - JS(p_s(\hat{x}), p_t(\hat{x}))$, where $p_s(\hat{x})$ and $p_t(\hat{x})$ are the outputs of student and teacher.

### Multi-Resolution Synthesis

[Yin et al.](https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html) propose a second extension of DeepInversion. They implemented a variant that synthesizes images at a smaller resolution for the first two-thirds of the optimization process. After that, the images are upscaled to their full size. Similarly, we optimize the images at a 112x112 resolution for first 50% of iterations and then at 224x224 for the remaining 50%. This allows us to speed up the synthesis as there are much less values that have to be optimized in the first half.

Other approaches for solving this challenge mostly incorporates a Generative Adversarial Network (GAN) that produces the training data. However, the GAN must be trained on data that is in the same domain as the target model. As we do not know from which domain the 18 samples come from, we chose not to implement a GAN-based approach.

### Evaluation

Various combinations of [hyperparameter values (line 263 - 274)](src/deep_inversion.py) have been tested during this challenge. We also compared DeepInversion against its adaptive and multi-resolution variants. We observed that no method consistently produces more realistic images than the other two approaches. Combining Adaptive DeepInversion with multi-resolution synthesis does not result in a significant improvement either. Optimizing for the target class 0 resulted mostly in images that contain eyes, hair, fur, feathers or faces of animals. In rare cases, we also recognized text, vehicles and human-like creatures. However, such images achieved a cosine similarity near the baseline of 0.664. We assume that most generated images originate from the ImageNet pretraining. In contrast, abstract images containing much less fine details achieved a score of 0.767. Despite the higher score, none of these images could be confidently identified as one of the 18 authorized samples. On the other hand, synthesizing for class 1 did not give us clear images at all. Most images consist of random colors and patterns. Rarely, images of vehicles or animals were generated. Extracted samples for both classes as well as the submitted abstract images are provided in the `assets` folder.