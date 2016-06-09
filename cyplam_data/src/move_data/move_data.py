import os
import paramiko
import glob


def move_file(filename, destdir):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect('172.20.0.204', username='ryco', password='faraday')
    except paramiko.SSHException:
        print "Connection Failed"
        quit()

    # stdin, stdout, stderr = ssh.exec_command("ls ")
    # tdin, stdout, stderr = ssh.exec_command("sudo dmesg")
    # for line in stdout.readlines():
    #     print line.strip()

    dirname, name = os.path.split(filename)
    destname = os.path.join(destdir, name)

    sftp = ssh.open_sftp()
    sftp.put(filename, destname)
    print 'File %s transfered to %s' % (filename, destname)

    ssh.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str,
                        default=None,
                        help='path to input file')
    args = parser.parse_args()

    filename = args.file

    if filename is None:
        dirname = '/home/panadeiro/bag_data/'
        filenames = sorted(glob.glob(os.path.join(dirname, '*.bag')))
        filename = filenames[-1]
    move_file(filename, '/home/ryco/data/')
