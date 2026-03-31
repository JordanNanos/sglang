#!/bin/bash
#
### update the package
set -x
pushd .

sudo apt update
sudo apt install libibumad-dev rdma-core ibverbs-utils infiniband-diags -y
sudo apt install linux-headers-"$(uname -r)" libelf-dev  -y
sudo apt install gcc make libtool autoconf librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils  perftest ethtool libibverbs-dev rdma-core strace -y

BCM_DRIVER="bcm_234.1.124.0b.tar.gz"

## compile and install
cp /root/cache/${BCM_DRIVER} /tmp
cd /tmp
case "${BCM_DRIVER}" in
  *.zip)
    unzip -o ./${BCM_DRIVER}
    DIR_NAME="${BCM_DRIVER%.zip}"
    ;;
  *.tar.gz)
    tar zxf ./${BCM_DRIVER}
    DIR_NAME="${BCM_DRIVER%.tar.gz}"
    ;;
  *)
    echo "ERROR: unsupported archive format: ${BCM_DRIVER}"; exit 1
    ;;
esac
cd /tmp/${DIR_NAME}/drivers_linux/bnxt_rocelib
VERSION="${DIR_NAME#bcm_}"
BCM_LIB=$(ls -1 *.tar.gz)
tar zxf ${BCM_LIB}
cd "${BCM_LIB%.tar.gz}"
sh ./autogen.sh
sh ./configure
make -j8 2>&1 | tee log.make
make install 2>&1 | tee log.make.install


## clean up the inbox derivers
find /usr/lib64/ /usr/lib -name "libbnxt_re-rdmav*.so" -exec mv {} {}.inbox \;
sudo make install all
sudo sh -c "echo /usr/local/lib >> /etc/ld.so.conf"
sudo ldconfig
# sudo cp -f bnxt_re.driver /etc/libibverbs.d/

popd
