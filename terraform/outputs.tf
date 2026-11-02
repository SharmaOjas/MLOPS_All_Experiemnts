output "container_name" {
  value = docker_container.rul_api.name
}

output "api_url" {
  value = "http://localhost:${var.external_port}"
}

output "health_url" {
  value = "http://localhost:${var.external_port}/health"
}

output "image_id" {
  value = docker_image.rul_api.image_id
}
