from pathlib import Path
import numpy as np
import pefile
import struct
import torch
from torchvision.models import resnet18

# auxiliary function
# Convert virtual addresses shown in Ghidra to offset relative to image base
def convert_va_to_file_offset(pe, va, image_base):
    # rva = relative virtual address
    rva = va - image_base
    return pe.get_offset_from_rva(rva)

# Return null-terminated C string at a given virtual address
def get_string_at_va(pe, raw_bytes, image_base, va):
    # Compute file offset
    offset = convert_va_to_file_offset(pe, va, image_base)

    # Read string until null termination and return
    stop_idx = raw_bytes.index(b"\x00", offset)
    return raw_bytes[offset:stop_idx].decode("ascii")


def main():
    executable_path = Path("executable/classifier.exe")

    # Information found via Ghidra
    weights_offset = 0x13A00            # File offset, where the section ".weights" begins. Found by using the memory map
    model_size = 0x2AAC800              # Size of the section ".weights". Found by using the memory map
    weight_index_addr = 0x142AC4320     # Virtual address of the lookup table that maps a pointer of a weight name (e.g., "conv1.weight") to its offset and length inside ".weights"
    num_weights = 0x66                  # Found in the get_weight function (102 rows in weight_index)
    model = resnet18(num_classes=1)     # model architecture found in the build_and_execute function

    # Load executable for Windows 11
    pe = pefile.PE(str(executable_path))        
    raw_bytes = executable_path.read_bytes()    # raw bytes of executable
    image_base = pe.OPTIONAL_HEADER.ImageBase   # preferred memory address where Windows 11 should load the executable

    # Extract ".weights" section containing the actual weights of the DNN (we use "<f4" as each weight has a size of 4 bytes (float32))
    weights_section = np.frombuffer(raw_bytes[weights_offset:weights_offset + model_size], dtype="<f4")
    print("weights_section.shape: ", weights_section.shape)

    # Read weight_index table
    weight_index_offset = convert_va_to_file_offset(pe, weight_index_addr, image_base)
    rows = []

    for i in range(num_weights):
        # Each entry/cell is either a pointer or a uint64_t and thus has a size of 8 bytes. Each table row contains 3 entries --> 8*3=24 = 0x18
        row_offset = weight_index_offset + i * 0x18

        # Read row (we use "<QQQ" as we are reading 3 unsigned 64-bit integer)
        name_va, offset, len = struct.unpack_from("<QQQ", raw_bytes, row_offset)

        name = get_string_at_va(pe, raw_bytes, image_base, name_va)
        rows.append((name, offset, len))

    print("weight_index table: ")
    for r in rows:
        print(r)

    # Build ResNet-18 model
    state = model.state_dict()
    new_model_state = {}

    for name, offset, len in rows:
        target_shape = state[name].shape

        weight_arr = weights_section[offset:offset + len].copy()
        weight_tensor = torch.from_numpy(weight_arr).reshape(target_shape)

        if state[name].dtype in (torch.int64, torch.long):
            weight_tensor = weight_tensor.to(torch.int64)

        new_model_state[name] = weight_tensor

    # Load extracted weights into model and save model
    missing, unexpected = model.load_state_dict(new_model_state, strict=False)
    print("missing: ", missing)
    print("unexpected: ", unexpected)
    torch.save(model.state_dict(), "extracted_model.pth")

if __name__ == "__main__":
    main()