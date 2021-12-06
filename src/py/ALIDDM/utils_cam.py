from torch._C import device
import vtk
import os
import glob
from utils import PolyDataToTensors
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils_class import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, PointLights,
)
import csv


def dataset(data):
    model_lst = []
    landmarks_lst = []
    datalist = []
    normpath = os.path.normpath("/".join([data, '**', '']))
    for img_fn in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".vtk"]]:
            if True in ['Lower' in img_fn]:
                model_lst.append(img_fn)
        if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".json"]]:
            if True in ['Lower' in img_fn]:
                landmarks_lst.append(img_fn)

    # for i in model_lst:
    #     print("model_lst :",i)
    # for i in landmarks_lst:
    #     print("landmarks_lst :",i)
    
    # if len(model_lst) != len(landmarks_lst):
    #     print("ERROR : Not the same number of models and landmarks file")
    #     return
    
    # for file_id in range(0,len(model_lst)):
    #     data = {"model" : model_lst[file_id], "landmarks" : landmarks_lst[file_id]}
    #     datalist.append(data)
    
    # # for i in datalist:
    # #     print("datalist :",i)
    # # print(datalist)
    # return datalist
    
    outfile = os.path.join(os.path.dirname(data),'data_O.csv')
    fieldnames = ['surf', 'landmarks', 'number_of_landmarks']
    data_list = []
    for idx,path in enumerate(landmarks_lst):
        data = json.load(open(path))
        markups = data['markups']
        landmarks_dict = markups[0]['controlPoints']
        number_of_landmarks = len(landmarks_dict)

        rows = {'surf':model_lst[idx],
                'landmarks':path,
                'number_of_landmarks':number_of_landmarks }
        data_list.append(rows)
    
    with open(outfile, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)

    return outfile

def generate_sphere_mesh(center,radius,device):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(center[0],center[1],center[2])
    sphereSource.SetRadius(radius)

    # Make the surface smooth.
    sphereSource.SetPhiResolution(10)
    sphereSource.SetThetaResolution(10)
    sphereSource.Update()
    vtk_landmarks = vtk.vtkAppendPolyData()
    vtk_landmarks.AddInputData(sphereSource.GetOutput())
    vtk_landmarks.Update()

    verts_teeth,faces_teeth = PolyDataToTensors(vtk_landmarks.GetOutput())

    verts_rgb = torch.ones_like(verts_teeth)[None]  # (1, V, 3)
    # color_normals = ToTensor(dtype=torch.float32, device=self.device)(vtk_to_numpy(fbf.GetColorArray(surf, "Normals"))/255.0)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts_teeth], 
        faces=[faces_teeth],
        textures=textures)
    
    return mesh

def Training(epoch, agents, agents_ids,num_step, train_dataloader, loss_function, optimizer, device):
    for batch, (V, F, CN, LP) in enumerate(train_dataloader):
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )
        batch_loss = 0
        # img_batch = torch.empty((0)).to(device)

        for aid in agents_ids: #aid == idlandmark_id
            agents[aid].reset_sphere_center(V.shape[0])

            print('---------- agents id :', aid,'----------')

            NSteps = num_step
            aid_loss = 0
        
            agents[aid].trainable(True)
            agents[aid].train()

            for i in range(NSteps):
                print('---------- step :', i,'----------')

                optimizer.zero_grad()   # prepare the gradients for this step's back propagation

                x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]
                
                x += agents[aid].sphere_centers
                # print('coord sphere center :', agent.sphere_center)
                
                lm_pos = torch.empty((0)).to(device)
                for lst in LP:
                    lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)
                # print(lm_pos)
                
                loss = torch.sqrt(loss_function(x, lm_pos))

                loss.backward()   # backward propagation
                optimizer.step()   # tell the optimizer to update the weights according to the gradients and its internal optimisation strategy
                
                l = loss.item()
                aid_loss += l
                print("Step loss:",l)
                agents[aid].sphere_centers = x.detach().clone()
            
            aid_loss /= NSteps
            agents[aid].trainable(False)

            print(f"agent {aid} loss:", aid_loss)
            
            agents.writer.add_scalar('distance',aid_loss,epoch)

        #     batch_loss += aid_loss
        
        # batch_loss /= len(agents_ids)
        # writer.add_scalar('distance',batch_loss)
        
def Validation(epoch,agents,agents_ids,test_dataloader,num_step,loss_function,output_dir,early_stopping,device):
    with torch.no_grad():
        for batch, (V, F, CN, LP) in enumerate(test_dataloader):

            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            for aid in agents_ids: #aid == idlandmark_id
                print('---------- agents id :', aid,'----------')
                agents[aid].reset_sphere_center(V.shape[0])

                NSteps =  num_step
                aid_loss = 0
                epoch_loss = 0
                agents[aid].eval() 

                for i in range(NSteps):
                    print('---------- step :', i,'----------')

                    x = agents[aid](meshes)  #[batchsize,time_steps,3,224,224]

                    x += agents[aid].sphere_centers
                    # print('coord sphere center :', agent.sphere_center)
                    
                    lm_pos = torch.empty((0)).to(device)
                    for lst in LP:
                        lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)
                    
                    loss = torch.sqrt(loss_function(x, lm_pos))
                    print('agent position : ', x)
                    print('landmark position :', lm_pos)

                    l = loss.item()
                    aid_loss += l
                    print("Step loss:",l)
                    agents[aid].sphere_centers = x.detach().clone()
                    
                    
                aid_loss /= NSteps
                print("Step loss:", aid_loss)
                epoch_loss += aid_loss

                if aid_loss<agents[aid].best_loss:
                    agents[aid].best_loss = aid_loss
                    agents[aid].best_epoch_loss = epoch
                    torch.save(agents[aid].attention, os.path.join(output_dir, f"best_attention_net_{aid}.pth"))
                    torch.save(agents[aid].delta_move, os.path.join(output_dir, f"best_delta_move_net_{aid}.pth"))
                    print("saved new best metric network")
                    print(f"Model Was Saved ! Current Best Avg. Dice: {agents[aid].best_loss} at epoch: {agents[aid].best_epoch_loss}")
            
                
              

            epoch_loss /= len(agents_ids)
            
            early_stopping(epoch_loss)

            if epoch_loss<agents[aid].best_epoch_loss:
                torch.save(agents[0].features_net, os.path.join(output_dir, "best_feature_net.pth"))

            return early_stopping.early_stop

def affichage(data_loader,phong_renderer):
    for batch, (V, F, Y, F0, CN, IP,IL) in enumerate(data_loader):
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(
            verts=V,   
            faces=F, 
            textures=textures
        )
        
        agent = Agent(meshes,phong_renderer)
        list_pictures = agent.shot().to(device)
        for pictures in list_pictures:
            plt.imshow(pictures)
            plt.show()

def pad_verts_faces(batch):
    verts = [v for v, f, cn, lp in batch]
    faces = [f for v, f, cn, lp in batch]
    # region_ids = [rid for v, f, rid, fpid0, cn, ip, lp in batch]
    # faces_pid0s = [fpid0 for v, f, fpid0, cn, ip, lp in batch]
    color_normals = [cn for v, f, cn, lp in batch]
    # ideal_position = [ip for v, f, fpid0, cn, ip, lp in batch]
    landmark_position = [lp for v, f, cn, lp in batch]
    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), landmark_position

def SavePrediction(data, outpath):
    print("Saving prediction to : ", outpath)
    img = data.numpy()
    output = sitk.GetImageFromArray(img)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)

def pad_verts_faces_prediction(batch):
    verts = [v for v, f, cn, ma , sc in batch]
    faces = [f for v, f, cn, ma , sc in batch]
    color_normals = [cn for v, f, cn, ma , sc in batch]
    mean_arr = [ma for v, f, cn, ma , sc  in batch]
    scale_factor = [sc for v, f, cn, ma , sc in batch]

    return pad_sequence(verts, batch_first=True, padding_value=0.0), pad_sequence(faces, batch_first=True, padding_value=-1), pad_sequence(color_normals, batch_first=True, padding_value=0.), mean_arr, scale_factor

def Accuracy(agents,test_dataloader,agents_ids,min_variance,loss_function,writer,device):
    list_distance = ({ 'obj' : [], 'distance' : [] })
    with torch.no_grad():
        for batch, (V, F, CN, LP) in enumerate(test_dataloader):

            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            for aid in agents_ids: #aid == idlandmark_id
                print('---------- agents id :', aid,'----------')
                agents[aid].reset_sphere_center(V.shape[0])

                agents[aid].eval() 
                
                pos_center = agents[aid].search(meshes,min_variance) #[batchsize,3]
                
                lm_pos = torch.empty((0)).to(device)
                for lst in LP:
                    lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)  #[batchsize,3]
                
                
                for i in range(V.shape[0]):
                    loss = torch.sqrt(loss_function(pos_center[i], lm_pos[i]))
                    list_distance['obj'].append(str(aid))
                    list_distance['distance'].append(float(loss.item()))
                
                # writer.add_scalar('distance',loss)

            # print(list_distance)
        
        sns.violinplot(x='obj',y='distance',data=list_distance)
        plt.show()

def Prediction(agents,dataloader,agents_ids,min_variance):
    list_distance = ({ 'obj' : [], 'distance' : [] })
    groupe_data = {}

    with torch.no_grad():
        for batch, (V, F, CN, MR, SF) in enumerate(dataloader):

            textures = TexturesVertex(verts_features=CN)
            meshes = Meshes(
                verts=V,   
                faces=F, 
                textures=textures
            )
            
            for aid in agents_ids: #aid == idlandmark_id
                coord_dic = {}
                print('---------- agents id :', aid,'----------')
                agents[aid].reset_sphere_center(V.shape[0])

                agents[aid].eval() 
                
                pos_center = agents[aid].search(meshes,min_variance) #[batchsize,3]
                
                # lm_pos = torch.empty((0)).to(device)
                # for lst in LP:
                #     lm_pos = torch.cat((lm_pos,lst[aid].unsqueeze(0)),dim=0)  #[batchsize,3]
                
                # loss = loss_function(pos_center, lm_pos)

                # list_distance['obj'].append(str(aid))
                # list_distance['distance'].append(float(loss.item()))
                for i in range(V.shape[0]):
                    # print(pos_center[i],SF[i],MR[i])
                    scale_surf = SF[i]
                    mean_arr = MR[i]
                    landmark_pos = pos_center[i]
                    # print(landmark_pos,MR,scale_surf)

                    pos_center = (landmark_pos/scale_surf)- mean_arr
                    pos_center = pos_center.cpu().numpy()
                    # print(pos_center)
                    coord_dic = {"x":pos_center[0],"y":pos_center[1],"z":pos_center[2]}
                    groupe_data.append({f'Lower_O-{aid+1}':coord_dic})


            print(list_distance)
        
        print("all the landmarks :" , groupe_data)
    
    return groupe_data

def GenControlePoint(groupe_data):
    lm_lst = []
    false = False
    true = True
    id = 0
    for landmark,data in groupe_data.items():
        id+=1
        controle_point = {
            "id": str(id),
            "label": landmark,
            "description": "",
            "associatedNodeID": "",
            "position": [data["x"], data["y"], data["z"]],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": true,
            "locked": true,
            "visibility": true,
            "positionStatus": "preview"
        }
        lm_lst.append(controle_point)

    return lm_lst

def WriteJson(lm_lst,out_path):
    false = False
    true = True
    file = {
    "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
    "markups": [
        {
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "locked": false,
            "labelFormat": "%N-%d",
            "controlPoints": lm_lst,
            "measurements": [],
            "display": {
                "visibility": false,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.0],
                "selectedColor": [1.0, 0.5000076295109484, 0.5000076295109484],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": false,
                "pointLabelsVisibility": true,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
                "useGlyphScale": true,
                "sliceProjection": false,
                "sliceProjectionUseFiducialColor": true,
                "sliceProjectionOutlinedBehindSlicePlane": false,
                "sliceProjectionColor": [1.0, 1.0, 1.0],
                "sliceProjectionOpacity": 0.6,
                "lineThickness": 0.2,
                "lineColorFadingStart": 1.0,
                "lineColorFadingEnd": 10.0,
                "lineColorFadingSaturation": 1.0,
                "lineColorFadingHueOffset": 0.0,
                "handlesInteractive": false,
                "snapMode": "toVisibleSurface"
            }
        }
    ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(file, f, ensure_ascii=False, indent=4)

    f.close

# def Prediction(agents,load_model,datas):
#     agents.load_state_dict(torch.load(load_model,map_location=device))
#     with torch.no_grad():
#         print("Loading data from :", args.dir)
#                 for image in img_model:
#                     new_image = torch.from_numpy(image).permute(2,0,1) # convertion in tensor (7,258,258)
#                     img_output = net(new_image)
#                     # print(torch.from_numpy(img_output).size())
#                     output = torch.cat(img_output,0)
#             output = torch.cat(img_output,0)
#             distance = loss_function(img_output, IP)
#             print('difference between exact and predict position :', distance)
#             list_distance.append(distance)
        
#     SavePrediction(output, output_path)