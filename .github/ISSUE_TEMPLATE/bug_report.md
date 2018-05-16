---
name: Bug report
about: Create a report to help improve this package

---

**Note about Euler angles**
Before filing any bug report that involves Euler angles please read [this page](https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible), especially the [last section](https://github.com/moble/quaternion/wiki/Euler-angles-are-horrible#opening-issues-and-pull-requests) which details how to open a bug report involving Euler angles.  In short, you need to understand what the code is doing.  Just because the code did something different from what you expected, that doesn't mean the code is wrong; more likely your expectation was wrong.

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Provide a minimal code example that allows this bug to be reproduced.  If relevant, fill out the code sample below with something that demonstrates your problem.
```python
import numpy as np
import quaternion

print(quaternion.z * quaternion.x)
```

**Expected behavior**
A clear and concise description of what you expected to happen and why.

**Environment (please complete the following information):**
 - OS, including version
 - Installation method (conda, pip, or compiled from source)
 - Numpy version (use `np.version.version`)
 - Quaternion version (use `quaternion.__version__`)

**Additional context**
Add any other context about the problem here.
