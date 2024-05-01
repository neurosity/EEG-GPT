# WARN: UNDER ACTIVE DEVELOPMENT

# Intro

## What is this?

This is a project to build a foundation model for EEG data. It is based on the NeuroGPT model by Wenhui Cui et al.

## What is EEG?

EEG is short for Electroencephalography. It is a non-invasive method of measuring brain activity. It is used to detect brain signals that are not easily captured by other methods, such as seizures and emotional states.

## Who is Neurosity?

Neurosity is a technology company that specializes in creating brain-computer interfaces. They have developed a device called the [Crown](neurosity.co), which is a wearable EEG headset that can measure brain activity. The data collected by the Crown can be used for a variety of applications, including mental health monitoring, cognitive enhancement, and controlling devices with your mind. Neurosity's mission is to empower individuals with the ability to understand and enhance their mental state.

# About this Repo

## Standardizing EEG

1. The model allows any electrode found in the 10-10 or 10-20 system.
2. The model expects 128.0 Hz sampling rate

# Based on

## NeuroGPT

[NeuroGPT](https://github.com/wenhui0206/NeuroGPT/blob/main/src/batcher/downstream_dataset.py) by Wenhui Cui et al.

```bibtex
@misc{cui2024neurogpt,
      title={Neuro-GPT: Towards A Foundation Model for EEG},
      author={Wenhui Cui and Woojae Jeong and Philipp Thölke and Takfarinas Medani and Karim Jerbi and Anand A. Joshi and Richard M. Leahy},
      year={2024},
      eprint={2311.03764},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Neurosity Foundational Model

[JeremyNixon/neurosity)](https://github.com/JeremyNixon/neurosity)

```bibtex
@misc{neurosity_eeg_dataset,
  title={Neurosity EEG Dataset},
  author={Nixon, Jeremy and Keller, AJ},
  year={2024},
  url={https://github.com/JeremyNixon/neurosity}
}
```
