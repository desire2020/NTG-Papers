# Paper Collection of Neural Text Generation (NTG)

Neural Text Generation refers to a kind of methods that mainly attempt to use NN as function approximators to mimic the underlying distribution of (natural) languages. The most important applications of the conditional version of this topic include Neural Machine Translation (NMT), neural image captioning and dialogue system (chatbot). However, the researches of NTG usually refer to those focus on the unconditional problem, that is to really learn the latent distribution of the target language (instead of a transformation mapping from source form to target form).

This repository presents a collection of previous research papers of Neural Text Generation (NTG), as well as a taxonomy constructed according to publication time, method paradigm or paper type.

# Taxonomy of Papers 

## Survey and Theoretical Analysis
* [Neural Text Generation: Past, Present and Beyond](https://arxiv.org/abs/1803.07133)
* [How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?](https://arxiv.org/abs/1511.05101)
## Metrics, Toolbox and Dataset
* [BLEU: a method for automatic evaluation of machine translation](https://dl.acm.org/citation.cfm?id=1073135)
* [METEOR: An automatic metric for MT evaluation with improved correlation with human judgments](http://www.aclweb.org/anthology/W05-0909)
* [Perplexity—a measure of the difficulty of speech recognition tasks](http://adsabs.harvard.edu/abs/1977ASAJ...62Q..63J)
* NLL<sub>oracle</sub>, NLL<sub>test</sub>, SelfBLEU, Texygen [Texygen: A Benchmarking Platform for Text Generation Models](https://arxiv.org/abs/1802.01886)
## Online-available Course
* [Introduction to Reinforcement Learning by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
## Research Paper
* Architechture
    * NNLM [A neural probabilistic language model](http://www.jmlr.org/papers/v3/bengio03a.html)
    * RNNLM [Recurrent neural network based language model](https://www.isca-speech.org/archive/interspeech_2010/i10_1045.html)
    * LSTM [Long short-term memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735)
    * GRU [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)
    * SRU [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)
    * Hierarchical Softmax [Classes for fast maximum entropy training](https://ieeexplore.ieee.org/abstract/document/940893/)
    * Feudal-like Language Model [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624)
* Training Algorithm / Models
    * Likelihood Based
        * Maximum Likelihood Estimation / Teacher Forcing [A learning algorithm for continually running fully recurrent neural networks](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.2.270)
        * Scheduled Sampling [Scheduled sampling for sequence prediction with recurrent neural networks](http://papers.nips.cc/paper/5956-scheduled-sampling-for-sequence-prediction-with-recurrent-neural-networks)
    * Static-target Reinforcement Learning
        * Policy Gradient [Policy gradient methods for reinforcement learning with function approximation](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
            * PG-BLEU: Use Policy Gradient to optimize BLEU.
    * Adversarial Methods
        * Adversary as a Regularization
            * Professor Forcing [Professor forcing: A new algorithm for training recurrent networks](http://papers.nips.cc/paper/6098-professor-forcing-a-new-algorithm-for-training-recurrent-networks)
        * Direct
            * SeqGAN [ SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](http://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14344/14489)
            * MaliGAN [Maximum-likelihood augmented discrete generative adversarial networks](https://arxiv.org/abs/1702.07983)
            * RankGAN [Adversarial ranking for language generation](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation)
            * ScratchGAN - Big Version of SeqGAN gets rid of pre-training! [Training Language GANs from Scratch](https://arxiv.org/abs/1905.09922)
            * RelGAN [RelGAN: Relational Generative Adversarial Networks for Text Generation](https://openreview.net/forum?id=rJedV3R5tm)
        * Adversarial Feature Matching
            * TextGAN [Adversarial feature matching for text generation](https://arxiv.org/abs/1706.03850)
        * Denoise Sequence-to-sequence Learning
            * MaskGAN [MaskGAN: Better Text Generation via Filling in the _](https://arxiv.org/abs/1801.07736)
        * Reparametrized Sampling
            * GSGAN [Gans for sequences of discrete elements with the gumbel-softmax distribution](https://arxiv.org/abs/1611.04051)
        * Learning to Exploit Leaked Information
            * LeakGAN [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624)
        * Smoothing-N-Rediscretization
            * WGAN-GP, GAN-GP [Adversarial generation of natural language](https://arxiv.org/abs/1705.10929)
    * Cooperative Methods
        * CoT [CoT: Cooperative Training for Generative Modeling of Discrete Data](https://arxiv.org/abs/1804.03782)
