# Trajectory references

These CPU fp32 recordings use the GSTRAJ2 format. Their scene arrays and compressed state payloads match
[the reference dataset at revision 5d576214](https://huggingface.co/datasets/Genesis-Intelligence/snapshots/tree/5d576214be3df0db6864e12524a37e19c6eaa333/rigid/__snapshots__/test_serialization).
Qualified type names identify the scene classes and solver fields. Authoring aliases excluded from
serialization are represented by their resolved quaternions and textures.

The SHA256 checksums of the state payloads, including their chunk headers, are:

| Environments | SHA256 |
| --- | --- |
| 0 | c71892a73b0fc4a9c09a160fba7c10432a0848a343c4b61be45b9cb08968b95f |
| 2 | 480f0235b2ef149a9b82c54203a38c33ab19a25edc833ab87d7fd5a75d376a38 |
