# Strength reduction on Neural Networks

This is the source code repository of [Strength Reduction Techniques in Compilers for Optimizing Inference on Edge Devices](https://www.diva-portal.org/smash/record.jsf?dswid=9537&pid=diva2%3A1940306&c=1&searchType=SIMPLE&language=en&query=alessio+petruccelli&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all).

## How to use
The `train` folder is where the model have been trained. We used Tensorflow, Keras and TFLite.
Structure of the folder
```
train
├── data
├── models
└── src
    ├── cnn.py
    ├── data_preprocessing
    ├── dataset.py
    ├── datasets
    ├── ds_cnn.py
    ├── __init__.py
    ├── logger
    ├── mlp.py
    └── tflite_utils.py
```

1. Install the required tool (pyenv is recommended)
```
pip install -r requirements.txt
```
2. Run a model, teh all have the same flags. I.e.
```
usage: mlp.py [-h] [-l LOAD] [-t] [-e] [--summary] [--lite LITE] [--name NAME]

CLI utility that enables training and evaluation. The model is always saved.

options:
  -h, --help            show this help message and exit
  -l LOAD, --load LOAD  Load a saved model
  -t, --train           Train the model (creates one or use the loaded one)
  -e, --eval            Evaluate the model and show the first 10 predictions.
  --summary             Show the model summary
  --lite LITE           Convert to TFLite and evaluate it. Available modes:[dyn,float16,int]
  --name NAME           Model name (default: )
```
3. Once the model is trained, under the `models` folder you can find the `.keras` model. If you have enabled to generate the *tflite* format, this will be under `model/lite/`.

The `inference` folder has this structure.
```
inference
├── emulation_template
├── extract_from_tflite.py
├── generator
├── gen.py
├── output.log
└── tests
```

Since the flatbuffers used by TFLITE where not the perfect format to work with, we have the `extract_from_tflite.py` script to parse the flatbuffers and extract in a JSON format a subset of the model informations.
Usage:
```
usage: extract_from_tflite.py [-h] [-m MODEL_PATH] [-o OUTPUT_PATH]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model-path MODEL_PATH
                        The name of the model to extract
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        The name of the file to save the extracted
```

Once the model is extracted, we can generate the code for our emulation:

1. Run the `gen.py` script to load the json file with the model information. This recreates the model and will perform one inference on a selected test image. Under the output folder, we are going to find the `include` and `src` folder with the C code.
```
usage: gen.py [-h] [-i INPUT] [-o OUTPUT] [-d {mnist,speech}]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        JSON extracted from TFLite
  -o OUTPUT, --output OUTPUT
                        Generates the C code for the emulation
  -d {mnist,speech}, --dataset {mnist,speech}
                        Dataset for which the model was trained
```
2. Now that we have our C source code, we can copy in the `emulation_template` the two folders. We did not generate a `main.c` file, this must be written by hand. It looks like something like this
```c
#ifdef DEBUG
#include "model.h"
#else
#include "model_opt.h"
#endif

#if defined(__AVR)
#include <avr/io.h>
#elif defined(__MSP430__) || defined(__arm__) || defined(__riscv)
#else
#include <stdio.h>
#endif

#include "macros.h"

static int argmax(const int8_t* arr, const size_t size)
{
    int max = 0;
    int8_t max_val = 0;
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max = i;
        }
    }
    return max;
}

// Pre-allocate the arrays
int8_t feature_maps_0[1][26][26][4] = { 0 };
int8_t maxpooled_0[1][13][13][4] = { 0 };
int8_t feature_maps_1[1][12][12][2] = { 0 };
int8_t maxpooled[72] = { 0 };
int8_t result[10] = { 0 };

int main()
{
    byte_t image = {
        {
#include "../image_8.dat"
        }
    };
    quantize(&image);
    conv2d_0((int8_t(*)[28][28][1])image.i8, feature_maps_0);
    max_pool2d_0(feature_maps_0, maxpooled_0);
    conv2d_1(maxpooled_0, feature_maps_1);
    max_pool2d_1(feature_maps_1, (int8_t(*)[6][6][2])maxpooled);
    fully_connected((int8_t(*)[72])maxpooled, result);
    int inference = argmax(result, 10);
    return inference;
}
```
3. Now we want to run the emulations. First of all you will need the different compilers for AVR, ARM, RISCV and MSP430.
> we suggest to take the one from TexasInstruments
You can find most of these compilers from your package-manager, i.e.
```
sudo pacman -S avr-gcc
```
> We built the RISC-V compiler from source as the one available was giving some problems.

Once the compilers are installed and set in your path (you might need to adjust the Makefiles)
```
make <arch>[arm,msp430,x86,riscv,avr]
```
This command will compile 4 different binaries, two with the standard implementation of the neural network with the `-Os` and `-O3` optimization flags, and two with strength-reduced implementation with the same flags respectively.

## Emulation
Before the emulation, be sure that also the `gdb` binaries for each architecture have been downloaded with the compiler. Before start emulating, in order to reporduce the results.
1. Clone the QEMU repository locally
```
git clone https://gitlab.com/qemu-project/qemu.git --branch v8.2.4
```
2. Apply the patch with the `memcnt` plugin we developed.
```
git apply qemu.patch
```
3. Follow its README instruction to build the binary.
4. Once is build, to use the repo as it is, create a symbolic link of the binary inside `emulation_template` and call it `qemu-bin`
5. To run teh emulation, run the bash script `bash emulate-[arch].sh model.elf` with the compiled binary as `elf` format, in this way it's possible to keep the debug information. This will start the QEMU machine and will stop the processor. Now, use GDB to attach to the GDB server hosted by QEMU and start the execution. You can start and terminate the execution using the gdb scripts inside the `gdb` folder.
```
gdb -x [arch]-qemu.gdb
```
6. The results from the plugins will be available in the emulation folder as txt files.