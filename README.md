# tbug
testing framework for debugging tensors across different runs and projects

# install
`git clone https://github.com/dnbt777/tbug/`

# usage

## logging

you must first log the tensors during a run. to do this, add the following to your code:

```
import tdiff

...


tdiff.capture(tensor, name="section_name.child_name", project="project_name_or_run_id_here")
```

you need to not do compilation for this to work (i.e. jit if in jax, or pytorch.compile if using pytorch)

## analysis

see simple_analysis_example.ipynb
