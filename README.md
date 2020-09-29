
---

<div align="center">    
 
# MNIST Pytorch Lightning

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   

First, install dependencies   

``` bash
# clone project   
git clone https://github.com/congvm-it/mnist.git

# install project   
cd mnist
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd mnist

# run module (example: mnist as your main contribution)   
python main.py    
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

``` python
  from datasets import MNISTDataModule
  pl.seed_everything(1234)

  parser = ArgumentParser()
  parser = pl.Trainer.add_argparse_args(parser)
  parser = MNISTDataModule.add_argparse_args(parser)
  parser.add_argument('--num_workers', default=4, type=int)
  args = parser.parse_args()

  # Dataloader
  data_loader = MNISTDataModule(num_workers=args.num_workers)

  # Model
  model = Classifier(**vars(args))

  # Training
  trainer = pl.Trainer(gpus=args.gpus, 
                        max_epochs=2, 
                        limit_train_batches=200)
  trainer.fit(model, data_loader)

  # Testing
  trainer.test(model, test_dataloaders=data_loader.test_dataloader())
```

### Citation   

```
@article{YourName, 
  title={Your Title}, 
  author={Your team}, 
  journal={Location}, 
  year={Year}
}
```   
