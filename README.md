# AiFi sorting with DNNs challenge Aug 20-23, 2021
## Discussion and related academic work
### Problem statement
    * Sorting is a classical computer science thoeritical algorithmic problem with widely used applications
    * Can we apply deep learning to solve this problem better than popular methods such as quicksort? What might these approaches look like?
    * What are the pros and cons of different approaches that we can try using deep learning?
### The naive approach -> 1 using RNN based auto-encoder and decoder models (sequence to sequence models)
    * The RNNs can be directly applied to this problem as we have inputs and outpus as sequences.
    * The RNNs are auto-encoder decoder networks which have a hidden state vector between each sequential block and the overall probability distribution over most optimal output sequence states is learnt.
    * Here, the input sequence order does not really matter as it can be considered a set
    * We have to force the RNN to unlearn this input sequence order by making it see all possible permutations for a particular output permutation. This is computationally expensive.
    * Also, the network cannot generalize well for longer length sequences as it requires deliberate retraining.
    * RNNs by themselves even specialized variants (LSTMs, GRUs) are super slow and unstable to train and are prone to vanishing gradient, overfitting problems when dealing with longer sequences.
### The naive approach -> 2 using content based attention (Neural Turning Machine paper)
    * Auto-encoder, decoder networks with an addition of attention layers to encode better context of the input space and thereby make the task of decoding easier to learn given the context over the entire input sequence instead of info from just one hidden state.
    * In other words, our decoder can access a weighted average of  DNNs become more stable and scalable to train for longer sequences.
    * But we still have the problem of not generalizing well to longer sequences during inference time as well as explicit guidance to not pay attention to the input sequence order
### Pointer Networks approach
    * Pointer Networks are a brilliant design particluarly for dealing with input sets to generate output sequences.
    * They generalize well to the longer length sets although they had not seen them during training
    * Here the attention mechanism is modeled as a probability distribuition as a pointer to the input items.
    * We still face the limitations of scale as any variant of RNNs need a lot of compute power.
    * The authors say this approach can be applied to the problem of sorting numbers however, in practice this is much harder if you just increase the sequence length to 10+.
    * The authors address the problem of sorting numbers with another paper called Order matters (sequence to sequence models with sets)
### Order Matters: Sequence to Sequence models with sets
    * Here the authors mention that input sequence order matters for solving such problems with DNNs
    * They devise a strategy to replace just the lstm in the encoder with a feed forward network + lstm, the decoder is still a pointer network
    * They also propose a read, process, write and reproduce method.
    * However, even with this approach, the sorting performance for sequence length greater than 10 suffers a lot.
### NNsort
    * Here the authors propose an iterative algorithm which sorts numbers with multiple passes instead of a single shot.
    * They use merge sort based divide and conquer approach to deal with conflicting sub-sequences
    * They claim in average case scenario, this provides a significant speedup over quicksort.
### Machine learning sort O(N)?
    * The authors claim that if the dataset size is sufficiently large, we can assume a smoother distribuition. and therefore, they make use of this knowledge to do a distribuition aware sort
    * This approach is promising but might only work on large and natural datasets

## Included code and usage
### Getting Started
    * Install a conda environment by following the official instructions
    * Activate an environment and then proceed with the following commands
    ```
    conda create --clone base --name aifi
    conda activate aifi
    conda install cudatoolkit=10.0 -c conda-forge
    python -m pip install keras==2.0.6 tensorflow-gpu==1.15
    ```

### Description
    * Train a PointerNetwork and sort a given input sequence of a given length
    * The model performance is very bad and suffers from a lot of duplicate pointers.
    * Please feel free to add or contribute for better performance


### Example usage training and test_sequence help
    ```
    cd pointer-networks-experiments
    $ python train_sort_numbers.py --help
    Using TensorFlow backend.
    usage: train_sort_numbers.py [-h] [--n_examples N_EXAMPLES]
                                 [--upper_limit UPPER_LIMIT] [--epochs EPOCHS]
                                 [--test_sequence TEST_SEQUENCE [TEST_SEQUENCE ...]]
                                 n_steps

    Train LSTM_encoder+PointerLSTM_decoder for sorting numbers

    positional arguments:
      n_steps               Sequence length (recommended: 5)

    optional arguments:
      -h, --help            show this help message and exit
      --n_examples N_EXAMPLES
                            n_examples (recommended: 10000)
      --upper_limit UPPER_LIMIT
                            upper_limit of the input data (recommended: 10)
      --epochs EPOCHS       no_of_epochs to be trained for (recommended: 10)
      --test_sequence TEST_SEQUENCE [TEST_SEQUENCE ...]
                            test_sequence to view predicted output sequence
    ```

### Example usage:
    * Here the upper_limit is for the model to generalize well (for example if the upper limit is only 5 for seq_len 5. It will only see permutations of whole numbers up to 5)
    * But if the upper_limit is 10 for the seq_len 5. It will see permutations of whole numbers of upto 10 but only length 5 sequences. Therefore, the network generalizes well
    ```
    cd pointer-networks-experiments
    python train_sort_numbers.py 5 --upper_limit 10 --epochs 10 --test_sequence 2 1 6 8 9
    ```

## Accuracy discussion
    * The best model is modeled after the order matters: Sequence to Sequence models with sets paper
    * The model only reaches 66% validation accuracy if we train for 1000 epochs with seq_len 5 and upper_limit 10
    * Meaning the model saturates at 66% accuracy if the upper_limit is greater than seq_len
    * The model accuracy when querying beyond the training upper limit is around 40 % with seq_len 5
    * If we fix the upper limit to be 5 during training and testing and query only a permutation with seq_len 5, then we get 100% accuracy.
    * This means that we have to explicitly and painfully train all the possible permutations for the model to learn well indicating that this is computationally exponential task.

### Code bootstrapped from [https://github.com/zygmuntz/pointer-networks-experiments](https://github.com/zygmuntz/pointer-networks-experiments)
    * Related blogpost -> [http://fastml.com/introduction-to-pointer-networks/](http://fastml.com/introduction-to-pointer-networks/)

### Improvements that can be applied for the future
    * Transformers architecture can be leveraged and improved by using some techniques described by Performers paper, as well (generalized attention kernels approach) as CAiT (residual connection based approach) paper.
    * Adaptive Computation time for LSTMs used in the encoder stage
    * NNSort in combination with Pointer Network approach (we might need to go deeper and train longer)

## References:
* https://jacobjinkelly.github.io/2018/06/20/sorting-numbers-with-a-neural-network/
* Machine Learning sort [arXiv:1805.04272v2 [cs.LG] 15 Aug 2018](https://arxiv.org/pdf/1805.04272.pdf)
* NN Sort [arXiv:1907.08817v3 [cs.DS] 24 Dec 2019](https://arxiv.org/pdf/1907.08817.pdf)
* Pointer Networks [arXiv:1506.03134v2 [stat.ML] 2 Jan 2017](https://arxiv.org/pdf/1506.03134.pdf)
* https://github.com/zphang/adaptive-computation-time-pytorch
* https://github.com/keon/pointer-networks
* https://github.com/Guillem96/pointer-nn-pytorch
* http://fastml.com/introduction-to-pointer-networks/

## Additional reading:
* https://openaccess.thecvf.com/content_CVPR_2019/html/Engilberge_SoDeep_A_Sorting_Deep_Net_to_Learn_Ranking_Loss_Surrogates_CVPR_2019_paper.html
* http://vana.kirj.ee/public/proceedings_pdf/2020/issue_3/proc-2020-3-186-196.pdf
* [Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision](https://arxiv.org/pdf/2105.04019.pdf)
* Overcoming catastrophic forgetting in neural networks [arXiv:1612.00796v2 [cs.LG] 25 Jan 2017](https://arxiv.org/pdf/1612.00796.pdf)
* Adaptive Computation Time for Recurrent Neural Networks [arXiv:1603.08983v6 [cs.NE] 21 Feb 2017](https://arxiv.org/pdf/1603.08983.pdf)
* https://distill.pub/2016/augmented-rnns/#adaptive-computation-time