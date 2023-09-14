# How to Use MiNER

## Running with Provided Datasets

The default dataset to run is the BioCreative V, just run those command one after the others to train and test on this dataset.

```sh
./scripts/pretrain_miner.sh
```

```sh
./scripts/train_partial_ner.sh
```

```sh
./scripts/test_partial_ner.sh
```

```sh
./scripts/train_smooth_ner.sh
```

```sh
./scripts/test_smooth_ner.sh
```

If you want to use an other one of the provided datasets, go into the `scripts` folder and change the necessary pathes to the ones needed.


## Running with Custom Datasets

You'll need:
* A raw `txt` file where each line represent a text.
* A `txt` file listing all possible labels in IOB format, starting by "O".
* `txt` dictionnaries organised the same way as for the other datasets (e.g., one ditionary = one entity type with an explicit name, and one gazetteer per line).
* `txt` dictionary anme "UNK.txt" with a list of your own quality potential phrases from the raw `txt` file. You could use [AutoNER](https://github.com/shangjingbo1226/AutoNER) to get them for example.

Once you have all the necessary documents, go into the scripts folder and change the differents paths accordingly.

You can now run the same commands as shown in the previous part.
