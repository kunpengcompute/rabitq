# Release Notes

## Version Mapping

### Product Version

<a name="table62675726"></a>
<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p48427257"><a name="p48427257"></a><a name="p48427257"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p16166112734513"><a name="p16166112734513"></a><a name="p16166112734513"></a><span id="text1726192733514"><a name="text1726192733514"></a><a name="text1726192733514"></a>26.0.RC1</span></p>
</td>
</tr>
<tr id="row5497143514612"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p162251517551"><a name="p162251517551"></a><a name="p162251517551"></a>Software Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p51757141375"><a name="p51757141375"></a><a name="p51757141375"></a>RaBitQ</p>
</td>
</tr>
<tr id="row615762416269"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.4.1"><p id="p12158152417260"><a name="p12158152417260"></a><a name="p12158152417260"></a>Software Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.4.1 "><p id="p51757141375"><a name="p51757141375"></a><a name="p51757141375"></a>v1.0.0</p>
</td>
</tr>
</tbody>
</table>

### OS, Compiler, and CPU

| OS| CPU| Memory| Compiler|
|---------|--------|------|--------|
| openEuler 24.03 LTS SP3 | Kunpeng 950 processors| 24\*64GB | GCC 12.3.1 / LLVM 16.0.6 |
| Debian 12 | Kunpeng 950 processors| 24\*64GB | GCC 12 / LLVM 16.0.6 |

## Fixed Bugs

This is the first official release, and there are no historical defect fixes.

## Known Issues and Constraints

| No.| Description|
|------|------|
| 1 | Two patches (equivalence/non-equivalence) are mutually exclusive; you must choose one.|
| 2 | AArch64 NEON optimization is exclusive to the AArch64 architecture; while the code can still be compiled and run on x86_64, ARM-specific optimizations will be disabled.|
