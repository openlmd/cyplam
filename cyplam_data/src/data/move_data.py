import yaml
import paramiko


def test_connection(accessfile):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        with open(accessfile, 'r') as f:
            accessdata = yaml.load(f)
        ssh.connect(accessdata['IP'],
                    username=accessdata['user'],
                    password=accessdata['password'],
                    timeout=5)
        connection = True
    except:
        connection = False
    ssh.close()
    return connection


def move_file(accessfile, filename, destdir):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        with open(accessfile, 'r') as f:
            accessdata = yaml.load(f)
        ssh.connect(accessdata['IP'],
                    username=accessdata['user'],
                    password=accessdata['password'],
                    timeout=5)
        # stdin, stdout, stderr = ssh.exec_command("ls")
        # tdin, stdout, stderr = ssh.exec_command("sudo dmesg"
        # for line in stdout.readlines():
        #     print line.strip()
        dirname, name = os.path.split(filename)
        destname = os.path.join(destdir, name)

        sftp = ssh.open_sftp()
        sftp.put(filename, destname)
        print 'File %s transfered to %s' % (filename, destname)
    except paramiko.SSHException:
        print "Connection Failed"
    ssh.close()


if __name__ == "__main__":
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', type=str, default=None, help='input file')
    args = parser.parse_args()

    filename = args.file

    dirname = '/home/panadeiro/bag_data/'
    accessfile = os.path.join(dirname, 'access.yalm')

    if filename is None:
        filenames = sorted(glob.glob(os.path.join(dirname, '*.bag')))
        filename = filenames[-1]

    print 'Test:', test_connection(accessfile)
    move_file(accessfile, filename, '/home/ryco/data/')
