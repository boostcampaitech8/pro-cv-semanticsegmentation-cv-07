# Hand Bone Image Segmentation

## Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/tenaan">
        <img src="https://github.com/tenaan.png" width="100px;" alt=""/>
        <br />
        <sub><b>안태현</b></sub>
      </a>
      <br />
      <a href="https://github.com/tenaan" title="Code"></a>
    </td>
    <td align="center">
      <a href="https://github.com/jjeongbin0826">
        <img src="https://github.com/jjeongbin0826.png" width="100px;" alt=""/>
        <br />
        <sub><b>최정빈</b></sub>
      </a>
      <br />
      <a href="https://github.com/jjeongbin0826" title="Code"></a>
    </td>
    <td align="center">
      <a href="https://github.com/chocobanana20">
        <img src="https://github.com/chocobanana20.png" width="100px;" alt=""/>
        <br />
        <sub><b>이승현</b></sub>
      </a>
      <br />
      <a href="https://github.com/chocobanana20" title="Code"></a>
    </td>
    <td align="center">
      <a href="https://github.com/wooqi00">
        <img src="https://github.com/wooqi00.png" width="100px;" alt=""/>
        <br />
        <sub><b>윤종욱</b></sub>
      </a>
      <br />
      <a href="https://github.com/wooqi00" title="Code"></a>
    </td>
    <td align="center">
      <a href="https://github.com/Jun00511">
        <img src="https://github.com/Jun00511.png" width="100px;" alt=""/>
        <br />
        <sub><b>박준영</b></sub>
      </a>
      <br />
      <a href="https://github.com/Jun00511" title="Code"></a>
    </td>
  </tr>
</table>

## Project Overview

본 프로젝트는 Hand Bone X-ray 이미지를 활용하여 뼈 영역을 정밀하게 분할하는 Bone Segmentation 모델 개발을 목표로 한다. 이를 통해 골절 진단, 수술 계획 수립, 맞춤형 의료 장비 제작 등 의료 현장에서의 진단 정확도와 업무 효율성 향상에 기여하고자 한다.

## Performance

<table>
  <thead>
    <tr>
      <th>Setting</th>
      <th>UperNet</th>
      <th>HRNet</th>
      <th>SwinUNet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Base</td>
      <td>0.9487</td>
      <td>0.9642</td>
      <td>0.9447</td>
    </tr>
    <tr>
      <td>Encoder</td>
      <td>0.9493</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Augmentation</td>
      <td>0.9545</td>
      <td>0.9706</td>
      <td>0.9575</td>
    </tr>
    <tr>
      <td>Loss</td>
      <td>0.9557</td>
      <td>0.9709</td>
      <td>0.9545</td>
    </tr>
    <tr>
      <td>Scheduler</td>
      <td>0.9561</td>
      <td>0.9709</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Resize</td>
      <td>0.9743</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>TTA</td>
      <td>0.9745</td>
      <td>0.9753</td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan="4" align="center">
        <b>Ensemble (Best): 0.9758</b>
      </td>
    </tr>
  </tbody>
</table>

**Public**  
<img src="assets/public.png" width="600"/>

**Private**  
<img src="assets/private.png" width="600"/>
