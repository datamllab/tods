import paramiko


ssh = paramiko.SSHClient()
ssh.connect('datalab5.engr.tamu.edu', username='jiazhen.yu', password='123789zjY!@#$%')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('ls')

# print(ssh_stdout)