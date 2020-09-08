# Environment variables describing runtime environment.
# From inside Docker container it is not really possible to obtain
# information about the Docker image used for the container. This
# is why we use environment variable to pass this information in.
# See descriptions of "base_docker_image" and "docker_image" metadata.
D3M_BASE_IMAGE_NAME = 'D3M_BASE_IMAGE_NAME'
D3M_BASE_IMAGE_DIGEST = 'D3M_BASE_IMAGE_DIGEST'
D3M_IMAGE_NAME = 'D3M_IMAGE_NAME'
D3M_IMAGE_DIGEST = 'D3M_IMAGE_DIGEST'

# Limits on CPU and memory compute resources available to the runtime
# can be communicated also through environment variables because it is
# not always easy to determine them from inside limited environment
# that not all resources visible are also available.
# Should be in Kubernetes units or equivalent.
# See: https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/#meaning-of-cpu
#      https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/#meaning-of-memory
D3M_CPU = 'D3MCPU'
D3M_RAM = 'D3MRAM'

# Used by pipeline resolver to configure where to search for files with pipelines.
PIPELINES_PATH = 'PIPELINES_PATH'
