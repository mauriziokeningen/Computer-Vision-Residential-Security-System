import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../../../components/ui/card';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { Switch } from '../../../components/ui/switch';
import { Badge } from '../../../components/ui/badge';
import { Select, NativeSelect, SelectItem } from '../../../components/ui/select';

export function SettingsPanel() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      <Card>
        <CardHeader>
          <CardTitle>General</CardTitle>
          <CardDescription>Preferencias del sistema</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">Notificaciones push</div>
              <div className="text-xs text-slate-500">Enviar a residentes y guardias</div>
            </div>
            <Switch defaultChecked />
          </div>
          <div>
            <Label>Zona horaria</Label>
            <Select defaultValue="America/Mexico_City">
              <NativeSelect>
                <SelectItem value="America/Mexico_City">America/Mexico_City</SelectItem>
                <SelectItem value="UTC">UTC</SelectItem>
              </NativeSelect>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Reconocimiento facial</CardTitle>
          <CardDescription>Umbrales y calidad</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Umbral de similitud</Label>
            <Input type="number" defaultValue={0.52} step="0.01" />
            <p className="text-xs text-slate-500">Mayor = más estricto</p>
          </div>
          <div className="space-y-1">
            <Label>Normalización</Label>
            <Select defaultValue="ArcFace">
              <NativeSelect>
                <SelectItem value="ArcFace">Hiperesfera (ArcFace)</SelectItem>
                <SelectItem value="CosFace">CosFace</SelectItem>
              </NativeSelect>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Análisis corporal</CardTitle>
          <CardDescription>Acciones y ventanas</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1">
            <Label>Ventana temporal (frames)</Label>
            <Input type="number" defaultValue={64} />
          </div>
          <div className="space-y-1">
            <Label>Clases monitoreadas</Label>
            <div className="flex flex-wrap gap-1">
              {['Empujón', 'Golpe', 'Caída', 'Cuchillo', 'Pistola'].map((c) => (
                <Badge key={c} variant="outline" className="rounded-full">
                  {c}
                </Badge>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

