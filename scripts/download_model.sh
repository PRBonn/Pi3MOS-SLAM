#!/bin/bash

wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip && unzip models.zip
mv dpvo.pth checkpoints/dpvo.pth
rm models.zip
