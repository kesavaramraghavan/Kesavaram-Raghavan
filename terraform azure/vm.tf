resource "azurerm_resource_group" "testing" {
  name     = "example-resources"
  location = "West US"
}

data "azurerm_platform_image" "example" {
  location  = azurerm_resource_group.testing.location
  publisher = "Debian"
  offer     = "debian-11"
  sku       = "11"
}

resource "azurerm_virtual_network" "testing" {
  name                = "example-network"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.testing.location
  resource_group_name = azurerm_resource_group.testing.name
}

resource "azurerm_subnet" "testing" {
  name                 = "internal"
  resource_group_name  = azurerm_resource_group.testing.name
  virtual_network_name = azurerm_virtual_network.testing.name
  address_prefixes     = [cidrsubnet(azurerm_virtual_network.testing.address_space[0],8,2)]
}

resource "azurerm_public_ip" "testing" {
  name                = "testing"
  resource_group_name = azurerm_resource_group.testing.name
  location            = azurerm_resource_group.testing.location
  allocation_method = "Static"
  sku               = "Basic"
}

resource "azurerm_network_interface" "testing" {
  name                = "example-nic"
  location            = azurerm_resource_group.testing.location
  resource_group_name = azurerm_resource_group.testing.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.testing.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id = azurerm_public_ip.testing.id
  }
}

resource "azurerm_linux_virtual_machine" "testing" {
  name                  = "example-machine"
  resource_group_name   = azurerm_resource_group.testing.name
  location              = azurerm_resource_group.testing.location
  size                  = "Standard_A2_v2"
  admin_username        = "testing"
  network_interface_ids = [azurerm_network_interface.testing.id]

  admin_ssh_key {
    username   = "testing"
    public_key = file("D:/git/Kesavaram-Raghavan/terraform azure/id_rsa.pub")
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = data.azurerm_platform_image.example.publisher
    offer     = data.azurerm_platform_image.example.offer
    sku       = data.azurerm_platform_image.example.sku
    version   = data.azurerm_platform_image.example.version
  }
}
