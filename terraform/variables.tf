variable "docker_host" {
  description = "Docker daemon endpoint. Windows Docker Desktop default shown; Linux/macOS: unix:///var/run/docker.sock"
  type        = string
  default     = "npipe:////./pipe/dockerDesktopLinuxEngine"
}

variable "image" {
  description = "Locally built prediction API image"
  type        = string
  default     = "rul-api:1.0.0"
}

variable "container_name" {
  description = "Name of the Terraform-managed API container"
  type        = string
  default     = "rul-api-terraform"
}

variable "external_port" {
  description = "Host port for the API (8000 is used by Docker Compose)"
  type        = number
  default     = 8001

  validation {
    condition     = var.external_port > 1024 && var.external_port < 65536 && var.external_port != 8000
    error_message = "Use a free port above 1024 other than 8000 (Docker Compose)."
  }
}
