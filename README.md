**N.B.:** This repository is out of date. A reference implementation of Optimal Transport divergences for shape registration is now available on the [global-divergences](https://github.com/jeanfeydy/global-divergences) repository.

# lddmm-ot
MICCAI 2017 Paper - Optimal Transport for Diffeomorphic Registration

Authors :
Jean Feydy, Benjamin Charlier, F.-X. Vialard, Gabriel Peyré

This repositery contains three independent implementations of the Optimal Transport
data attachment term introduced in the paper :

- A simplistic "one-script" theano+python implementation, located in the 
  'Simple_script/' folder. It implements curves matching with a simplistic
  'curve length' measure embedding.

- A fully-fledged theano+python toolbox, located in the "LDDMM_Python" folder.
  It implements embeddings in the varifold space for curves and surfaces,
  and was used for Figure 1 (Protozoa).

- A fully-fledged Matlab toolbox, placed in the 'matlab/' directory.

It also hosts a mini implementation in Pytorch, that we will strive to make as
memory-efficient as possible. Hopefully, we can find a way to relieve the major
bottleneck of autodiffs libraries, as of 2017.

--------------------------------------------------------------------------------
Instructions for the python implementations
--------------------------------------------------------------------------------
You will need to install the autodiff library Theano on top of Jupyter/IPython.
N.B. : To use your Nvidia GPU, don't forget to put the appropriate settings in
your .theanorc (http://www.deeplearning.net/software/theano/tutorial/using_gpu.html).
For our experiments, it was simply :

```markdown
[nvcc]
flags=-D_FORCE_INLINES


[global]
device=cuda
floatX=float32
exception_verbosity=high
```

(With an Nvidia GeForce GTX 960M)

We advise the interested reader to get introduced to our implementations with
the 'Simple_script/' toy demo.

Then, to recompute the Figure 1, please :
- go into the 'LDDMM_Python/' folder.
- there, open a terminal and launch the 'jupyter notebook' command - provided by
  a fresh Anaconda install for instance.
- Your web browser should open itself. Through the web interface, go into
  'demo/Notebooks/Feb2017_paper/'.
- Click on 'curves_matching.ipynb' to open the IPython/Jupyter Notebook.
- Simply make a 'Cell/Run all'.
- Once the computation is over, you can run the 'pvpython render.py'
  (provided by the software Paraview) to convert the .vtk into the appropriate
  .png images.

The runs made to compute the presented images have been saved in
'run_1.html' and 'run_2.html' (the first one had crashed due to a garbage
collection problem combined to the super-greedy theano scan, we discarded
the error message to preserve anonymity).
You may check energy descent and computation time here.

N.B. : you may change the source and target by editing the
'demo/Notebooks/Feb2017_paper/data/source.png' and 'target.png' images.

As of today, theano's implementation of the 'scan' or 'for loop' operation,
is extremely memory-greedy : we can't use autodiff with 
~1000 sinkhorn iterations with more than 200 points without a memory overflow.
A future version of the toolbox will use the 'scan_checkpoints' routine 
(which allows one to set the speed/memory tradeoff), and eventualy allow us
to test our method on Eulerian image registration.


--------------------------------------------------------------------------------
Instructions for the matlab toolbox
--------------------------------------------------------------------------------
-----------
Quick start
-----------

A) Use with pure matlab code
	1) simplyrun the various examples in ./Script/	
	 
B) Use with cuda mex files (mandatory with surfaces)
	1) compile the mex files in ./Bin/kernels/ with the makefile.sh
	2) run the various examples in ./Script/ 

You  will be able to read the .vtk files with Paraview.



Best regards,

The authors.










