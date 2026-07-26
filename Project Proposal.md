# Motivation and Past Research

Generating synthetic baby faces with controlled emotional expressions
addresses critical gaps in child psychology research, healthcare
applications, and forensic tools, where diverse and ethically sourced
infant facial data is urgently needed but difficult to obtain due to
privacy regulations and data scarcity [@falkenberg2024child].
Furthermore, synthetic data enables the development of robust facial
expression recognition systems for early childhood mental health
assessment while circumventing the ethical challenges of collecting real
infant faces [@falkenberg2024child]. Despite recent advances in
text-to-image generative models, significant gaps remain
[@habib2024exploring].

Traditional GAN-based approaches, such as the Fake Face Generator by
@mahiuddin2022fake, produce noisy outputs, lack fine-grained expression
control, require extensive training, and rely on adult face datasets.
More recent methods like ChildDiffusion [@farooq2025childdiffusion]
employ Stable Diffusion with DreamBooth and LoRA to generate child
faces, but focus on broad attributes like ethnicity and aging rather
than fine-grained emotional expressions. Similarly, HDA-SynChildFaces
[@falkenberg2024child] uses GANs for child face recognition
benchmarking, prioritizing identity over expression generation and aging
adults down rather than generating faces from scratch.

While these works provide a foundation, none specifically address the
unique challenges of generating baby faces with controlled expressions.
Our project bridges this gap by fine-tuning a Stable Diffusion pipeline
with DreamBooth and LoRA on a limited dataset of 1,000 baby face images,
enabling precise control over happy, sad, and mad expressions and
offering an ethical, practical solution for synthetic baby face
generation.

# Method

The project will use an infant facial expression dataset containing
happy, crying, and angry expressions divided into training, validation,
and test/reference sets. A StyleGAN2-ADA baseline will be trained on the
training set, while the proposed SDXL model will be fine-tuned using
DreamBooth and LoRA.

During fine-tuning, DreamBooth adapts the pretrained model to the infant
expression domain while preserving its generative capabilities, and LoRA
efficiently updates only a small set of low-rank parameters within the
U-Net, reducing computational and memory requirements. After training,
both models will generate synthetic images for each target expression.
The SDXL model will generate images conditioned on text prompts such as
a happy infant," a crying infant," and "an angry infant," while
StyleGAN2-ADA will generate an equivalent number of images for fair
comparison.

Diffusion models and GANs generate images using fundamentally different
approaches. A diffusion model, such as SDXL, learns to gradually remove
noise from random data to generate images, allowing it to model complex
image distributions and leverage text conditioning to control attributes
such as happy" or crying" expressions. In contrast, a GAN, such as
StyleGAN2-ADA, uses a generator and discriminator in a competitive
process, where the generator learns to produce realistic images that can
fool the discriminator.

While GANs are capable of generating sharp and visually realistic
images, they may have limitations in diversity and precise semantic
control. Diffusion models, especially large-scale models such as SDXL,
benefit from greater model capacity and large-scale image-text
pretraining, enabling them to generate more diverse and semantically
aligned images. These differences in learning objectives, conditioning
mechanisms, and representation learning can lead SDXL and StyleGAN2-ADA
to produce different synthetic baby faces.

SDXL contains approximately 3.5 billion parameters, while the
StyleGAN2-ADA generator and discriminator contain approximately 65
million combined parameters. Because LoRA updates only a small portion
of SDXL's parameters, the total model parameter count and the number of
trainable parameters will both be reported.

::: {#tab:model-parameters}
  **Model**        **Approximate Parameters**        **Training Strategy**
  --------------- ---------------------------- ---------------------------------
  SDXL                    3.5 billion           DreamBooth and LoRA fine-tuning
  StyleGAN2-ADA            65 million             Training on infant dataset

  : Approximate model parameter counts.
:::

To make the comparison between StyleGAN2-ADA and SDXL fair, both models
will be trained or fine-tuned on the same infant facial expression
dataset using the same training, validation, and test split and
preprocessing pipeline. The generated images will be evaluated using the
same metrics, including FID, FER accuracy, and CLIP Score, using the
same number of generated samples and the same real infant reference
distribution.

For StyleGAN2-ADA, the model will be trained using the infant dataset
with adaptive augmentation to improve generalization on the limited
data. For SDXL, the pretrained model will be fine-tuned using DreamBooth
and LoRA with the same dataset while keeping the base model and
fine-tuning strategy consistent across experiments. However, because
SDXL and StyleGAN2-ADA differ significantly in architecture, parameter
count, conditioning mechanisms, and pretraining, the comparison will be
interpreted as a comparison of overall generative performance rather
than an isolated comparison of model size.

# Metrics and Evaluation

Based on our proposed training setup, the total expected compute
requirement is 22--43 NVIDIA A100 GPU-hours. The StyleGAN2-ADA baseline
is estimated to require 8--15 GPU-hours. The SDXL LoRA fine-tuning stage
is expected to require approximately 12--24 GPU-hours. Image generation
and evaluation are expected to require an additional 2--4 GPU-hours.
Since model training may require debugging, failed runs, or repeated
experiments across random seeds, a practical compute budget would be
closer to 30--60 A100 GPU-hours.

For evaluation, our study will focus on three metrics: FID, CLIP Score,
and FER performance. FID will be used to compare the distribution of
generated images to the distribution of real infant images from a
reference set. A lower FID score indicates that the generated images are
more realistic in terms of visual features.

The CLIP Score will measure how well each generated image matches its
text prompt, such as a happy infant face" or a crying infant face." A
higher CLIP Score indicates that the image is more closely aligned with
the intended prompt.

Finally, FER performance will be used to evaluate whether the generated
images clearly represent the intended facial expression. A pretrained
facial expression recognition model will classify each generated image,
and higher FER accuracy will indicate that the model produces images
that more consistently preserve the target emotion.

## Mathematical Definitions of Evaluation Metrics

### Fréchet Inception Distance

Fréchet Inception Distance (FID) measures how similar feature
distributions are between real and generated images [@heusel2017gans].
Images are passed through a pretrained Inception-v3 network, and the
feature representations are modeled as multivariate Gaussian
distributions. FID is calculated:

$$\begin{equation}
\mathrm{FID}
=
\left\lVert \boldsymbol{\mu}_{r}
-
\boldsymbol{\mu}_{g} \right\rVert_{2}^{2}
+
\operatorname{Tr}
\left(
\boldsymbol{\Sigma}_{r}
+
\boldsymbol{\Sigma}_{g}
-
2
\left(
\boldsymbol{\Sigma}_{r}
\boldsymbol{\Sigma}_{g}
\right)^{1/2}
\right),
\label{eq:fid}
\end{equation}$$

where $\boldsymbol{\mu}_{r}$ and $\boldsymbol{\Sigma}_{r}$ are the mean
and covariance matrix of the real-image features, and
$\boldsymbol{\mu}_{g}$ and $\boldsymbol{\Sigma}_{g}$ are the mean and
covariance matrix of the generated-image features. A lower FID means the
generated-image distribution is more similar to the real-image
distribution.

In addition to an overall FID score, class-conditional FID scores will
be calculated separately for each target expression. This will allow the
study to determine whether one model performs well overall but struggles
to generate a particular infant expression.

### CLIP Score

CLIP Score measures alignment between a generated image and the text
prompt used to describe its intended content
[@radford2021learning; @hessel2021clipscore]. Let $f_I(x_i)$ denote the
CLIP image embedding of generated image $x_i$, and let $f_T(t_i)$ denote
the CLIP text embedding of its corresponding prompt $t_i$. The cosine
similarity for an image--prompt pair is

$$\begin{equation}
s_i
=
\frac{
f_I(x_i)^{\top} f_T(t_i)
}{
\left\lVert f_I(x_i) \right\rVert_2
\left\lVert f_T(t_i) \right\rVert_2
}.
\label{eq:clip-cosine}
\end{equation}$$

Following the CLIPScore formulation, the score for a set of $N$
generated images is

$$\begin{equation}
\mathrm{CLIPScore}
=
\frac{1}{N}
\sum_{i=1}^{N}
2.5 \max(s_i, 0).
\label{eq:clip-score}
\end{equation}$$

A higher CLIP Score means stronger agreement between the generated image
and its prompt. Both models will be evaluated using the same frozen CLIP
model and the same prompt templates.

### Facial Expression Recognition Performance

Facial Expression Recognition (FER) will measure whether the visible
expression in a generated image matches the expression requested from
the generative model. Let $h_k(x_i)$ be the probability assigned by a
frozen FER classifier to expression class $k$ for generated image $x_i$.
The predicted expression is

$$\begin{equation}
\hat{y}_i
=
\arg\max_{k \in \{1,\ldots,C\}}
h_k(x_i),
\label{eq:fer-prediction}
\end{equation}$$

where $C$ is the number of expression classes. Overall FER accuracy is

$$\begin{equation}
\mathrm{FER}_{\mathrm{accuracy}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{I}
\left(
\hat{y}_i = y_i
\right),
\label{eq:fer-accuracy}
\end{equation}$$

where $y_i$ is the expression requested when image $x_i$ was generated,
and $\mathbb{I}(\cdot)$ is an indicator function equal to one when the
prediction is correct and zero otherwise.

Because the number of generated images may differ across expression
classes, macro-averaged FER accuracy will also be reported:

$$\begin{equation}
\mathrm{FER}_{\mathrm{macro}}
=
\frac{1}{C}
\sum_{c=1}^{C}
\left[
\frac{1}{N_c}
\sum_{i:y_i=c}
\mathbb{I}
\left(
\hat{y}_i = c
\right)
\right],
\label{eq:fer-macro}
\end{equation}$$

where $N_c$ is the number of generated images conditioned on expression
class $c$. A higher FER score indicates that the generated images more
consistently display the requested emotional expression.

# Hypothesis

- Fine-tuning Stable Diffusion XL (SDXL) using DreamBooth and LoRA on an
  infant facial expression dataset will generate more realistic and
  semantically aligned synthetic infant facial expression images than
  StyleGAN2-ADA.

- Due to its larger model capacity and more advanced diffusion-based
  architecture, SDXL is expected to learn richer representations of
  infant facial characteristics, resulting in improved image quality and
  expression fidelity.

- SDXL is expected to achieve:

  1.  Lower Fréchet Inception Distance (FID), indicating closer
      similarity between generated and real infant image distributions.

  2.  Higher Facial Expression Recognition (FER) performance, indicating
      more accurate expression generation.

  3.  Higher CLIP Score, indicating stronger alignment between generated
      images and text prompts.

# Reproducibility

For reproducibility, we will use version 3 of the Smart Baby Monitoring
System Y10 dataset from Roboflow Universe. The dataset will provide the
infant facial expression images used for training, validation, and
evaluation. The dataset is available at:

::: center
[Smart Baby Monitoring System Y10 Dataset, Version
3](https://universe.roboflow.com/smart-baby-monitorind/smart-baby-monitoring-system-y10/dataset/3)
:::

For the diffusion-based pipeline, we will use the pretrained Stable
Diffusion XL base checkpoint `stabilityai/stable-diffusion-xl-base-1.0`
and fine-tune the U-Net using LoRA while keeping the VAE and text
encoder frozen. For the GAN baseline, we will use NVIDIA's official
`NVlabs/stylegan2-ada-pytorch` implementation of StyleGAN2-ADA. We will
also make our models and supporting materials publicly available through
our GitHub and Hugging Face repositories.

The project work will be divided so that Divya Pimparkar is responsible
for training the GAN baseline model, Rahul Dey is responsible for
training the Stable Diffusion model, and Jessica Henson is responsible
for evaluating the generated images using the selected metrics.

## Team Channels {#team-channels .unnumbered}

[InfantEmotionGen
GitHub](https://github.com/JessicaHenson01/InfantEmotionGen)

[InfantEmotionGen Hugging Face](https://huggingface.co/InfantEmotionGen)

# Detailed Evaluation Protocol {#app:evaluation}

The evaluation procedure will be identical for SDXL and StyleGAN2-ADA.
Both models will use the same held-out real reference set, image
preprocessing pipeline, output resolution, and number of generated
images per expression class. Generated images will not be manually
filtered before evaluation.

## FID Evaluation

FID will compare generated images with held-out real infant images using
the same feature extractor and reference distribution for both models.
We will report an overall FID score and separate FID scores for each
expression class. Lower values will indicate greater similarity between
the real and generated image distributions.

## CLIP Score Evaluation

Each generated image will be paired with a fixed text prompt
corresponding to its intended expression, such as "a photo of a happy
infant." The same prompts and frozen CLIP model will be used for both
generative models. Scores will be averaged overall and within each
expression class, with higher values indicating stronger image--text
alignment.

## FER Evaluation

A fixed FER classifier will evaluate whether each generated image
displays its intended expression. Before it is applied to synthetic
images, its performance on held-out real infant images will be reported
to establish that it is a reliable evaluator. FER results will include
overall accuracy, per-class accuracy, macro-averaged accuracy, and a
confusion matrix.

## Metric Interpretation

Each metric measures a different property. FID evaluates distributional
realism, CLIP Score evaluates semantic alignment, and FER evaluates
expression fidelity. Therefore, model performance will be assessed using
all three metrics together rather than relying on a single score.
