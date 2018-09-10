### Torchpack
- [ ] Implemented 'get' of config dict
- [ ] Fix cfg bugs, None type attr
- [ ] Unit test with mmcv and torchpack


### MMDetection

#### Basic
- [ ] Training not on cluster
- [ ] Now we using args as a global flow. TOO UGLY!
- [ ] Default logger only with gpu0
- [ ] Instead print of logger


#### Testing
- [ ] Implement distributed Testing
- [ ] parallel test in torckpack review
- [ ] Single GPU Test


#### Reformat
- [ ] All param names should be re-considered
        such as rpn_train_cfg or rpn_train? proposal_list and proposals?
- [ ] functions in 'core' should be refactored, such as target function
- [ ] Merge single test & aug test as one function

#### New features
- [ ] Loss params go into config.
- [ ] Multi-head communication.
