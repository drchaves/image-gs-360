import re

with open('model.py', 'r') as f:
    content = f.read()

ws_loss_code = """
        h = images.shape[-2]
        y_coords = torch.arange(h, device=images.device, dtype=images.dtype)
        ws_weights = torch.cos((y_coords - h / 2.0 + 0.5) * (torch.pi / h)).view(1, h, 1)

        if self.l1_loss_ratio > 1e-7:
            l1_err = torch.abs(images - self.gt_images)
            self.l1_loss = self.l1_loss_ratio * (l1_err * ws_weights).mean()
            self.total_loss += self.l1_loss
        else:
            self.l1_loss = None

        if self.l2_loss_ratio > 1e-7:
            l2_err = F.mse_loss(images, self.gt_images, reduction='none')
            self.l2_loss = self.l2_loss_ratio * (l2_err * ws_weights).mean()
            self.total_loss += self.l2_loss
        else:
            self.l2_loss = None
"""

# Replace L1 loss logic
old_loss_code_pattern = r"        if self\.l1_loss_ratio > 1e-7:\n            self\.l1_loss = self\.l1_loss_ratio \* F\.l1_loss\(images, self\.gt_images\)\n            self\.total_loss \+= self\.l1_loss\n        else:\n            self\.l1_loss = None\n        if self\.l2_loss_ratio > 1e-7:\n            self\.l2_loss = self\.l2_loss_ratio \* F\.mse_loss\(images, self\.gt_images\)\n            self\.total_loss \+= self\.l2_loss\n        else:\n            self\.l2_loss = None"
content = re.sub(old_loss_code_pattern, ws_loss_code.strip(), content)

with open('model.py', 'w') as f:
    f.write(content)
