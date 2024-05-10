import gradio
import torch

from modules import devices
from modules import sd_samplers_cfg_denoiser
from modules import script_callbacks
from modules import scripts
from modules import shared
from modules.shared import opts
from modules.processing import StableDiffusionProcessing
import k_diffusion
from k_diffusion import sampling

_schedulers = ['karras', 'exponential', 'polyexponential', 'vp']
_p: StableDiffusionProcessing = None
_number_of_controls = 8
_enable_default = [False] * _number_of_controls
_enable = _enable_default
_use_raw_sigma_default = [False] * _number_of_controls
_use_raw_sigma = _use_raw_sigma_default
_scheduler_default = ['karras'] * _number_of_controls
_scheduler = _scheduler_default
_start_value_default = [999] * _number_of_controls
_start_value = _start_value_default
_end_value_default = [0] * _number_of_controls
_end_value = _end_value_default
_step_default = [50] * _number_of_controls
_step = _step_default
_rho_default = [7] * _number_of_controls
_rho = _rho_default

class Noise_scheduler_modifier(scripts.Script):
    def title(self):
        return 'Noise scheduler modifier'
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        global _number_of_controls, _enable_default, _use_raw_sigma_default, _scheduler_default, _start_value_default, _end_value_default, _step_default, _rho_default
        enable = []
        use_raw_sigma = []
        scheduler = []
        start_value = []
        end_value = []
        step = []
        rho = []
        with gradio.Accordion('Noise scheduler modifier', open=False) as accordion:
            for i in range(_number_of_controls):
                with gradio.Accordion(f'Control {i + 1}', open=False):
                    with gradio.Row():
                        enable.append(gradio.Checkbox(False, label=f'Enable control {i + 1}'))
                        use_raw_sigma.append(gradio.Checkbox(False, label=f'Use raw sigma {i + 1}'))
                        scheduler.append(gradio.Dropdown(_schedulers, value='karras', multiselect=False, label=f'Schaduler {i + 1}'))
                    with gradio.Row():
                        start_value.append(gradio.Number(999, label=f'Start value {i + 1}', step=0.000001))
                        end_value.append(gradio.Number(0, label=f'End value {i + 1}', step=0.000001))
                    with gradio.Row():
                        step.append(gradio.Number(50, label=f'Step {i + 1}', step=1))
                        rho.append(gradio.Number(7, label=f'rho {i + 1}', step=0.000001))
        self.infotext_fields =  [(accordion, lambda d: gradio.Accordion.update(open=True in d.get('Noise shceduler modifier enable', _enable_default)))] + \
                                [(enable[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier enable', _enable_default)[i])) for i in range(_number_of_controls)] + \
                                [(use_raw_sigma[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier use raw sigma', _use_raw_sigma_default)[i])) for i in range(_number_of_controls)] + \
                                [(scheduler[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier scheduler', _scheduler_default)[i])) for i in range(_number_of_controls)] + \
                                [(start_value[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier start value', _start_value_default)[i])) for i in range(_number_of_controls)] + \
                                [(end_value[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier end value', _end_value_default)[i])) for i in range(_number_of_controls)] + \
                                [(step[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier step', _step_default)[i])) for i in range(_number_of_controls)] + \
                                [(rho[i], lambda d: gradio.Dropdown.update(value=d.get('Noise shceduler modifier rho', _rho_default)[i])) for i in range(_number_of_controls)]
        return *enable, *use_raw_sigma, *scheduler, *start_value, *end_value, *step, *rho
    
    def process_batch(self, p:StableDiffusionProcessing, *args, **kwargs):
        global _p, _number_of_controls, _enable, _use_raw_sigma, _scheduler, _start_value, _end_value, _step, _rho
        _enable, _use_raw_sigma, _scheduler, _start_value, _end_value, _step, _rho = ([], [], [], [], [], [], [])
        for i in range(_number_of_controls):
            _enable.append(getattr(p, f'Noise_shceduler_modifier_enable_{i + 1}', args[0 * _number_of_controls + i]))
            _use_raw_sigma.append(getattr(p, f'Noise_shceduler_modifier_use_raw_sigma_{i + 1}', args[1 * _number_of_controls + i]))
            _scheduler.append(getattr(p, f'Noise_shceduler_modifier_scheduler_{i + 1}', args[2 * _number_of_controls + i]))
            _start_value.append(getattr(p, f'Noise_shceduler_modifier_start_value_{i + 1}', args[3 * _number_of_controls + i]))
            _end_value.append(getattr(p, f'Noise_shceduler_modifier_end_value_{i + 1}', args[4 * _number_of_controls + i]))
            _step.append(getattr(p, f'Noise_shceduler_modifier_step_{i + 1}', args[5 * _number_of_controls + i]))
            _rho.append(getattr(p, f'Noise_shceduler_modifier_rho_{i + 1}', args[6 * _number_of_controls + i]))
        if True in _enable:
            p.sampler_noise_scheduler_override = Noise_scheduler_modifier.sampler_noise_scheduler_override
            p.extra_generation_params['Noise shceduler modifier enable'] = _enable
            p.extra_generation_params['Noise shceduler modifier use raw sigma'] = _use_raw_sigma
            p.extra_generation_params['Noise shceduler modifier scheduler'] = _scheduler
            p.extra_generation_params['Noise shceduler modifier start value'] = _start_value
            p.extra_generation_params['Noise shceduler modifier end value'] = _end_value
            p.extra_generation_params['Noise shceduler modifier steps'] = _step
            p.extra_generation_params['Noise shceduler modifier rho'] = _rho
        _p = p

    def sampler_noise_scheduler_override(steps):
        global _p
        model_wrap_cfg = CFGDenoiserKDiffusion(_p.sampler)
        model_wrap = model_wrap_cfg.inner_model
        sigmas = None
        for i in range(4):
            if _enable[i]:
                _step[i] = int(_step[i])
                if not _use_raw_sigma[i]:
                    _start_value[i] = int(_start_value[i])
                    _end_value[i] = int(_end_value[i])
                    if _start_value[i] < 0 or _start_value[i] > 999 or _end_value[i] < 0 or _end_value[i] > 999:
                        raise IndexError('Specify value between 0 and 999 if "Use raw sigma" is unchecked.')
                if _scheduler[i] == 'karras':
                    if sigmas is None:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = sampling.get_sigmas_karras(_step[i], sigma_min, sigma_max, _rho[i], devices.device)
                    else:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = torch.cat((sigmas[:-1], sampling.get_sigmas_karras(_step[i], sigma_min, sigma_max, _rho[i], devices.device)))
                elif _scheduler[i] == 'exponential':
                    if sigmas is None:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = sampling.get_sigmas_exponential(_step[i], sigma_min, sigma_max, devices.device)
                    else:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = torch.cat((sigmas[:-1], sampling.get_sigmas_exponential(_step[i], sigma_min, sigma_max, devices.device)))
                elif _scheduler[i] == 'polyexponential':
                    if sigmas is None:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = sampling.get_sigmas_polyexponential(_step[i], sigma_min, sigma_max, _rho[i], devices.device)
                    else:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = torch.cat((sigmas[:-1], sampling.get_sigmas_polyexponential(_step[i], sigma_min, sigma_max, _rho[i], devices.device)))
                elif _scheduler[i] == 'vp':
                    if sigmas is None:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = sampling.get_sigmas_vp(_step[i], sigma_max, sigma_min, _rho[i], devices.device)
                    else:
                        sigma_min = _end_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_end_value[i]].item()
                        sigma_max = _start_value[i] if _use_raw_sigma[i] else model_wrap.sigmas[_start_value[i]].item()
                        sigmas = torch.cat((sigmas[:-1], sampling.get_sigmas_vp(_step[i], sigma_max, sigma_min, _rho[i], devices.device)))
                else:
                    raise RuntimeError("Unavailable scheduler option")
        return sigmas

class CFGDenoiserKDiffusion(sd_samplers_cfg_denoiser.CFGDenoiser):
    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = k_diffusion.external.CompVisVDenoiser if shared.sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
            self.model_wrap = denoiser(shared.sd_model, quantize=shared.opts.enable_quantization)

        return self.model_wrap
    
def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module

    def confirm_scheduler(p, xs):
        for x in xs:
            if x not in _schedulers:
                raise RuntimeError(f"Unknown Scheduler: {x}")
            
    extra_axis_options = []
    for i in range(_number_of_controls):
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] enabled {i + 1}', bool, xyz_grid.apply_field(f'Noise_shceduler_modifier_enabled_{i + 1}')))
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] use raw sigma {i + 1}', bool, xyz_grid.apply_field(f'Noise_shceduler_modifier_use_raw_sigma_{i + 1}')))
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] scheduler {i + 1}', str, xyz_grid.apply_field(f'Noise_shceduler_modifier_scheduler_{i + 1}'), confirm=confirm_scheduler, choices=lambda: _schedulers))
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] start value {i + 1}', float, xyz_grid.apply_field(f'Noise_shceduler_modifier_start_value_{i + 1}')))
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] end value {i + 1}', float, xyz_grid.apply_field(f'Noise_shceduler_modifier_end_value_{i + 1}')))
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] steps {i + 1}', int, xyz_grid.apply_field(f'Noise_shceduler_modifier_step_{i + 1}')))
        extra_axis_options.append(xyz_grid.AxisOption(f'[Noise scheduler modifier] rho {i + 1}', float, xyz_grid.apply_field(f'Noise_shceduler_modifier_rho_{i + 1}')))

    if not any("[Noise scheduler modifier]" in x.label for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
    try:
        make_axis_options()
    except Exception as e:
        print(e)

script_callbacks.on_before_ui(callback_before_ui)
