#

import subprocess

def runcmd(cmd, verbose = False, *args, **kwargs):
    #code from https: // www.scrapingbee.com / blog / python - wget /
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

runcmd("pip install tensorflow", True)
runcmd("pip install aif360", True)
runcmd("pip install aif360[AdversarialDebiasing]",True)
runcmd("pip install aif360[LawSchoolGPA]", True)
runcmd("pip install aif360[Reductions]", True)
runcmd("pip install pandas", True)