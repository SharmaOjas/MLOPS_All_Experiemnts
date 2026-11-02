# Terraform provisions the RUL prediction API as a standalone Docker container.
# It owns its own image tag, container name and host port, so it never touches
# the Docker Compose stack (port 8000) or the Kubernetes deployment (port 30080).

terraform {
  required_version = ">= 1.5"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.6"
    }
  }
}

provider "docker" {
  host = var.docker_host
}

# Uses the image built from the project's Dockerfile:
#   docker build -f docker/Dockerfile -t rul-api:1.0.0 .
# (The provider's own `build` block hangs against Docker Engine 29, so the image is
# built with the Docker CLI and Terraform manages the container lifecycle.)
# keep_locally = true: `terraform destroy` removes the container but never the shared image.
resource "docker_image" "rul_api" {
  name         = var.image
  keep_locally = true
}

resource "docker_container" "rul_api" {
  name    = var.container_name
  image   = docker_image.rul_api.image_id
  restart = "unless-stopped"

  ports {
    internal = 8000
    external = var.external_port
  }

  env = ["DEPLOYED_BY=terraform"]

  healthcheck {
    test     = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
    interval = "15s"
    timeout  = "3s"
    retries  = 3
  }

  labels {
    label = "managed-by"
    value = "terraform"
  }
}
