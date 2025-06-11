# tbug
<div style="text-align:center;">
  <img src="https://github.com/user-attachments/assets/8ad597a3-4605-4f29-b06b-8b1ab6018216" width=400 height=400></img>
  <p>debug tensors across different runs and projects</p>
</div>

# example
[simple example](https://github.com/dnbt777/tbug/blob/main/simple_analysis_example.ipynb)


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

you need to disable compilation for this to work (i.e. jit if in jax, or pytorch.compile if using pytorch)

## analysis

see [simple_analysis_example.ipynb](https://github.com/dnbt777/tbug/blob/main/simple_analysis_example.ipynb)
