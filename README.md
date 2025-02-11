# GENERator

Welcome to the Gener Series!

In this repository, we present GENERator, a collection of generative genomic foundation models utilizing the transformer decoder architecture, trained on expansive DNA datasets derived from the [RefSeq database](https://www.ncbi.nlm.nih.gov/refseq/). The extensive and diverse pre-training data endow the GENERator with enhanced understanding and generation capabilities across various organisms. Our evaluations demonstrate that the GENERator consistently achieves state-of-the-art performance across a wide spectrum of benchmarks, including [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks/tree/main), [NT tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), and our newly proposed [Gener tasks](https://huggingface.co/GenerTeam). 

Beyond benchmark performance, the GENERator adheres to the central dogma of molecular biology, accurately generating protein-coding DNA sequences that produce proteins structurally analogous to known families. Moreover, the GENERator showcases significant promise in sequence optimization, particularly in the design of promoter sequences that regulate gene activity during various biological stages, highlighting its potential for a series of biologically significant tasks. Our findings position the GENERator as a vital resource for genomic research and biotechnological advancement. By enhancing our capability to interpret and predict genomic sequences, the GENERator paves the way for profound improvements in our understanding of complex biological systems and the development of precise genomic interventions.





Welcome to the Gener Series!

In this repository, we present GENERator, a generative genomic foundation model featuring a context length of 98k base pairs and 1.2B parameters, trained on an expansive dataset comprising 386 billion base pairs of eukaryotic DNA. The extensive and diverse pre-training data endow the GENERator with enhanced understanding and generation capabilities across various organisms. Our evaluations demonstrate that the GENERator consistently achieves state-of-the-art performance across a wide spectrum of benchmarks, including [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks/tree/main), [NT tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised), and our newly proposed [Gener tasks](https://huggingface.co/GenerTeam). 

Beyond benchmark performance, the GENERator adheres to the central dogma of molecular biology, accurately generating protein-coding DNA sequences that produce proteins structurally analogous to known families. Moreover, the GENERator showcases significant promise in sequence optimization, particularly in the design of promoter sequences that regulate gene activity during various biological stages, highlighting its potential for a series of biologically significant tasks. Our findings position the GENERator as a vital resource for genomic research and biotechnological advancement. By enhancing our capability to interpret and predict genomic sequences, the GENERator paves the way for profound improvements in our understanding of complex biological systems and the development of precise genomic interventions.

For more technical details, please refer to our paper [GENERator: A Long-Context Generative Genomic Foundation Model](https://huggingface.co/GenerTeam). 





An overview of the GENERator is provided in Figure 1.



The technical detrails of the GENERator model can be found in the paper [GENERator: A Long-Context Generative Genomic Foundation Model](https://huggingface.co/GenerTeam). 


We are committed to promoting transparency and collaboration in research. To this end, all necessary materials to replicate this work—including data, code, and model weights, will be made fully open-source on the GenerTeam GitHub \href{https://github.com/GenerTeam}{\textcolor{blue}{page}}. We hope that this contribution will help to advance the development of the DNA language model community.


In this repository, you will find the following:

- `GENERator-eukaryote-1.2b-base`
- `GENERator-eukaryote-3b-base`
- `GENERator-prokaryote-1.2b-base` (comming soon)
- `GENERator-prokaryote-3b-base` (comming soon)
- `GENERator-unified-7b-base` (looking for sponsorship)

