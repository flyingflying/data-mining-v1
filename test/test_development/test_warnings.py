# -*- coding:utf-8 -*-
# Author: lqxu

import warnings

def output():
    
    class DeprecationWarning(Warning):
        pass 
    
    warnings.warn("hello, warnings module", category=DeprecationWarning, stacklevel=2)


if __name__ == "__main__":
    

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("ignore", DeprecationWarning)
        print(w)
        # Trigger a warning.
        output()
        print(w[0])

    output()
    
    warnings.filterwarnings(action="ignore")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        print(w)
        import pyLDAvis
        print(w[0])
