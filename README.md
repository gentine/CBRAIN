# CBRAIN
To create the tf records, run eg:
```
python ./dataLoad.py --input_names=TAP,QAP,,SHFLX,LHFLX,PS' --output_names=SPDT,SPDQ --convert_units=true --convo=true
```


Then, to run experiments:
```
python ./main.py --input_names=TAP,QAP,,SHFLX,LHFLX,PS' --output_names=SPDT,SPDQ --convert_units=true --convo=true --localConvo=true --hidden=64,64,64,64,64,64 --batch_size=128 --optim=adam --lr=1e-3 --frac_train=0.8 --log_step=500 --epoch=10  
```

for non convolutional you must generate the records also (it will make separate ones).