# tbug
testing framework for debugging tensors across runs and projects

# install
`git clone https://github.com/dnbt777/tbug/`

# usage

## logging

you must first log the tensors

to do this:

```
import tdiff

...


tdiff.capture(tensor, name="section_name.child_name", project="project_name_or_run_id_here")
```

## analysis

see simple_usage_example.ipynb
