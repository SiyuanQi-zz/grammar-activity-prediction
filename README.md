# ICCV2017 - Predicting Human Activities Using Stochastic Grammar.

## Files in the repository:
- cpp: grammar induction code. Instructions for running the code can be find here: https://github.com/SiyuanQi/madios
- matlab: scripts mainly used for skeleton processing and feature extraction.
- python: files for low-level classifiers and grammar inference.

## Important notes
- The `cpp` folder contains code for learning grammar. For parsing/inference, most grammar related code are in `python/grammarutils.py`.
- If you want to use the code for **future prediction**, you need to modify the nltk library according to the instructions [here](https://gist.github.com/SiyuanQi/a056f2c152aa4174e4c1feb1de46d8fd).

If you find this code useful, please cite our work with the following bibtex:
```
@inproceedings{qi2017predicting,
    title={Predicting Human Activities Using Stochastic Grammar},
    author={Qi, Siyuan and Huang, Siyuan and Wei, Ping and Zhu, Song-Chun},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2017}
}
```
