# To-do

- [ ] Compression (have to change attention mask, make explicitly size of history) (test)
- [ ] R_pred only sees past states and actions, perhaps take last as critic value (idk)
- [ ] Rewrite DT
- [ ] Vectorize env
- [ ] check out swinv2
- [ ] do the vitmae thing?
- [x] Covariance matrix
- [x] Decaying learning rate
- [x] Gaussian policy stdev (make product of model)
- [x] Getting random NaNs in loc (cause of lr)
- [x] Loop, check what Lookahead is
- [x] Make RTG be RTG/steps
- [x] `rtg_preds` are critic
