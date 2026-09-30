R=/tmp2/mzjiang_usersim/r0_vllm
ls -la $R | head
echo "=== serve.log tail"; tail -60 $R/serve.log | grep -vE "^\s*$" | cut -c1-260 | tail -40
echo "=== serve.sh"; cat $R/serve.sh | cut -c1-250
