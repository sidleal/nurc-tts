import kfp
from kfp import dsl
from kubernetes import client as k8s_client

@dsl.component
def syntacc_tts_task(n_gpus: int):
    comp = dsl.ContainerOp(
        name='nurc-tts-train-task',
        image='sidleal/nurc-tts-training:0.8',
        command=['python', 'recipes/multilingual/cml_yourtts/train_syntacc.py', '--config_path "/app/recipes/multilingual/cml_yourtts/config.json"'],
        #python3 recipes/multilingual/cml_yourtts/train_syntacc.py --config_path "/app/recipes/multilingual/cml_yourtts/config.json"
        #command=["torchrun", f"--nproc-per-node={n_gpus}", "distil-whisper-train.py"],
    ) 
    return comp

@dsl.pipeline(
    name='NURC TTS Training',
    description='NURC TTS.'
)
def nurc_tts_pipeline():
    gpu_limit = 1
    train_task = syntacc_tts_task(n_gpus=gpu_limit)
    #train_task.set_gpu_limit(gpu_limit)
    train_task.set_cpu_request('4')
    train_task.set_cpu_limit('4')
    train_task.set_memory_request('32Gi')
    train_task.set_memory_limit('32Gi')
    #train_task.container.resources.limits['nvidia.com/mig-2g.20gb'] = '1'
    train_task.container.resources.limits['nvidia.com/gpu'] = '1'
    #train_task.container.resources.limits.pop('nvidia.com/gpu', None)


    #from kubernetes.client import V1Affinity, V1NodeAffinity, V1NodeSelector, V1NodeSelectorTerm, V1NodeSelectorRequirement
    # affinity = V1Affinity(
    #     node_affinity=V1NodeAffinity(
    #         required_during_scheduling_ignored_during_execution=V1NodeSelector(
    #             node_selector_terms=[
    #                 V1NodeSelectorTerm(
    #                     match_expressions=[
    #                         V1NodeSelectorRequirement(
    #                             key="nvidia.com/gpu.product",
    #                             operator="In",
    #                             values=["NVIDIA-H100-80G-HBM3"]
    #                         )
    #                     ]
    #                 )
    #             ]
    #         )
    #     )
    # )
    # train_task.add_affinity(affinity)

    # train_task.node_selector = {
    #     #"nvidia.com/gpu.product": "NVIDIA-GeForce-RTX-4090"    
    #     "nvidia.com/gpu.product": "NVIDIA-H100-80G-HBM3"
    # }

    from kubernetes.client import V1Toleration

    train_task.add_toleration(
        V1Toleration(
            key='region',
            operator='Equal',
            value='pipeline',
            effect='NoExecute'
        )
    )

    empty_dir_volume = k8s_client.V1Volume(
        name='dshm',
        empty_dir=k8s_client.V1EmptyDirVolumeSource(
            medium='Memory', 
            size_limit='2Gi'
        )
    )
    train_task.add_volume(empty_dir_volume)
    empty_dir_volume_mount = k8s_client.V1VolumeMount(
        name='dshm',
        mount_path='/dev/shm'
    )
    train_task.add_volume_mount(empty_dir_volume_mount)

    existing_volume = dsl.PipelineVolume(pvc='sidleal-speech-training')
    train_task.add_pvolumes({
        '/opt/training': existing_volume
    })

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=nurc_tts_pipeline, 
        package_path='nurc_tts_pipeline_dist_full.yaml'
    )
    print("Pipeline compiled successfully to nurc_tts_pipeline_dist.yaml")