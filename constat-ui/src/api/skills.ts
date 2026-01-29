// Skills API calls (user-based, not session-based)

import { get, post, put, del } from './client'

export interface SkillInfo {
  name: string
  prompt: string
  description: string
  filename: string
  is_active: boolean
}

export interface SkillsListResponse {
  skills: SkillInfo[]
  active_skills: string[]
  skills_dir: string
}

export interface SkillContentResponse {
  name: string
  content: string
  path: string
}

// List all skills for the user
export async function listSkills(): Promise<SkillsListResponse> {
  return get<SkillsListResponse>('/skills')
}

// Create a new skill
export async function createSkill(
  name: string,
  prompt: string,
  description: string = ''
): Promise<SkillInfo> {
  return post<SkillInfo>('/skills', { name, prompt, description })
}

// Get skill content (raw YAML)
export async function getSkillContent(skillName: string): Promise<SkillContentResponse> {
  return get<SkillContentResponse>(`/skills/${encodeURIComponent(skillName)}`)
}

// Update skill content (raw YAML)
export async function updateSkillContent(
  skillName: string,
  content: string
): Promise<{ status: string; name: string }> {
  return put<{ status: string; name: string }>(
    `/skills/${encodeURIComponent(skillName)}`,
    { content }
  )
}

// Delete a skill
export async function deleteSkill(skillName: string): Promise<{ status: string; name: string }> {
  return del<{ status: string; name: string }>(`/skills/${encodeURIComponent(skillName)}`)
}

// Set active skills
export async function setActiveSkills(
  skillNames: string[]
): Promise<{ status: string; active_skills: string[] }> {
  return put<{ status: string; active_skills: string[] }>('/skills/active', {
    skill_names: skillNames,
  })
}

// Get combined skills prompt
export async function getSkillsPrompt(): Promise<{ prompt: string; active_skills: string[] }> {
  return get<{ prompt: string; active_skills: string[] }>('/skills/prompt')
}
